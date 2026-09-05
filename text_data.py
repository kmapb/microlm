# `paths` must be imported before `datasets`/`transformers`: it's what points
# their caches at the shared datasettes mount, and they read the environment at
# import time.
import paths  # noqa: F401  (imported for its import-time side effects)

import numpy as np
import torch
import datasets
import pytorch_lightning as pl
from transformers import AutoTokenizer

TOKENIZER = None

# Hub datasets used to be reachable by bare "canonical" names. That namespace is
# gone, and loading scripts along with it, so map the names this repo grew up
# with onto the repos that still serve the same bytes. Anything not listed here
# is passed through untouched.
DATASET_ALIASES = {
    "wikitext": "Salesforce/wikitext",
    "c4": "allenai/c4",
    "openwebtext": "Skylion007/openwebtext",
    "tiny_stories": "roneneldan/TinyStories",
    # deepmind/pg19 is script-based (dead on modern hub); this parquet
    # mirror serves the same books with train/validation/test splits.
    "pg19": "emozilla/pg19",
    "deepmind/pg19": "emozilla/pg19",
}

# Configs moved too: wikitext's `-v1` variants were the script's names for what
# the parquet repo calls `-raw-v1`.
CONFIG_ALIASES = {
    ("Salesforce/wikitext", "wikitext-103-v1"): "wikitext-103-raw-v1",
    ("Salesforce/wikitext", "wikitext-2-v1"): "wikitext-2-raw-v1",
}

# Script-only datasets with no parquet mirror. Fail loudly rather than let the
# hub's "repository not found" error stand in for an explanation.
RETIRED_DATASETS = {
    "bookcorpus": "no parquet mirror on the hub; try 'Salesforce/wikitext' or 'HuggingFaceFW/fineweb-edu'",
    "bookcorpus/bookcorpus": "no parquet mirror on the hub; try 'Salesforce/wikitext' or 'HuggingFaceFW/fineweb-edu'",
    "the_pile": "withdrawn; 'HuggingFaceFW/fineweb-edu' is the closest live stand-in",
}


def resolve_dataset(name, config=None):
    """Map a possibly-legacy (name, config) pair onto one the hub still serves."""
    if name in RETIRED_DATASETS:
        raise ValueError(f"dataset {name!r} is retired: {RETIRED_DATASETS[name]}")
    resolved = DATASET_ALIASES.get(name, name)
    config = CONFIG_ALIASES.get((resolved, config), config)
    return resolved, config


def _setup_tokenizer():
    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")


def _tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        _setup_tokenizer()
    return TOKENIZER


def tokenize(text, add_special_tokens=True):
    return _tokenizer()(text, add_special_tokens=add_special_tokens)['input_ids']


def vocabulary_size():
    # len() rather than len(.vocab): it counts any added special tokens too.
    return len(_tokenizer())


def sep_token_id():
    return _tokenizer().sep_token_id


def pad_token_id():
    return _tokenizer().pad_token_id


def encode(s, add_special_tokens=True, truncation=True, max_length=None):
    return _tokenizer()(s['text'],
                        add_special_tokens=add_special_tokens,
                        max_length=max_length,
                        truncation=truncation)


def decode(t):
    return _tokenizer().decode(t)


class PackedWindows(torch.utils.data.IterableDataset):
    """Concatenate-and-chunk: tokenize a stream of {'text': ...} rows, join
    documents with a separator token, and emit dense windows of exactly
    `window` tokens. No padding ever reaches the model, short rows cost
    nothing (they just extend the buffer), and long documents span windows
    instead of being truncated.

    Documents are joined with [SEP] and get no [CLS] -- chat-time prompts
    don't have one either, so training and inference see the same shape of
    stream. The final partial buffer of a split is dropped rather than padded.

    `encode_fn` is injectable so the packing logic is testable without the
    tokenizer (and therefore without the hub).
    """

    def __init__(self, rows, window, eos_id=None, encode_fn=None):
        self.rows = rows
        self.window = window
        self.eos_id = sep_token_id() if eos_id is None else eos_id
        self.encode_fn = encode_fn or \
            (lambda text: tokenize(text, add_special_tokens=False))

    def __iter__(self):
        buf = []
        for row in self.rows:
            ids = self.encode_fn(row['text'])
            if not ids:  # blank lines contribute nothing, not even a [SEP]
                continue
            buf.extend(ids)
            buf.append(self.eos_id)
            while len(buf) >= self.window:
                window, buf = buf[:self.window], buf[self.window:]
                yield {'input_ids': torch.tensor(window, dtype=torch.long),
                       'num_tokens': self.window}


# e.g., dataset = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train')
def load_dataset(name, config, split='train', streaming=True, shuffle=True, num_proc=16):
    name, config = resolve_dataset(name, config)
    if streaming:
        ds = datasets.load_dataset(name, config, split=split, streaming=True)
    else:
        ds = datasets.load_dataset(name, config, split=split, streaming=False, num_proc=num_proc)
    shuf = ds
    if shuffle:
        if streaming:
            shuf = ds.shuffle(buffer_size=8192)
        else:
            shuf = ds.shuffle()
    return shuf


def packed_path(name, config, split):
    """Where the pre-tokenized uint16 stream for (dataset, config, split)
    lives -- on the shared data root, so one pack serves every run."""
    safe = f"{name}_{config}_{split}".replace('/', '--')
    return paths.DATA_ROOT / 'packed' / f"{safe}.u16"


def _pack_rows(id_lists, out_path, sep_id):
    """Write id-lists as one flat uint16 stream, [SEP]-joined; skip empties.
    Writes to a temp file and renames, so a partial pack never looks done."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.tmp')
    total = 0
    with open(tmp, 'wb') as fh:
        for ids in id_lists:
            if not ids:
                continue
            arr = np.empty(len(ids) + 1, dtype=np.uint16)
            arr[:-1] = ids
            arr[-1] = sep_id
            fh.write(arr.tobytes())
            total += arr.size
    tmp.rename(out_path)
    return total


def prepack(name, config, split='train', num_proc=16, shuffle_seed=1337,
            force=False):
    """Tokenize a whole split once (parallel) into a packed uint16 file.
    Documents are shuffled with a fixed seed before packing, so the stream
    order is decorrelated but reproducible. Returns the file path; no-op if
    the pack already exists."""
    out = packed_path(name, config, split)
    if out.exists() and not force:
        return out
    tokzr = _tokenizer()
    assert len(tokzr) < 2 ** 16, "uint16 pack requires vocab < 65536"

    rname, rcfg = resolve_dataset(name, config)
    ds = datasets.load_dataset(rname, rcfg, split=split, streaming=False,
                               num_proc=num_proc)
    ds = ds.shuffle(seed=shuffle_seed)

    def enc(batch):
        return {'ids': tokzr(batch['text'],
                             add_special_tokens=False)['input_ids']}

    # Small map batches: rows can be entire books (megabytes of text), and
    # the default 1000-row batch across num_proc workers OOM-kills workers.
    tok = ds.map(enc, batched=True, batch_size=16, num_proc=num_proc,
                 remove_columns=ds.column_names, desc=f"tokenize {split}")

    def rows():
        for batch in tok.iter(batch_size=1000):
            yield from batch['ids']

    total = _pack_rows(rows(), out, tokzr.sep_token_id)
    print(f"packed {total/1e9:.2f}B tokens -> {out}")
    return out


class PackedFileWindows(torch.utils.data.Dataset):
    """Map-style dense windows over a packed uint16 token file. Being
    map-style (unlike the streaming pipeline) buys window-level shuffling
    and multi-worker loading for free."""

    def __init__(self, path, max_length, start_frac=0.0, end_frac=1.0):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.L = max_length
        n = len(self.data) // max_length
        self.lo = int(n * start_frac)
        self.hi = int(n * end_frac)

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        j = (self.lo + i) * self.L
        ids = torch.from_numpy(np.asarray(self.data[j:j + self.L],
                                          dtype=np.int64))
        return {'input_ids': ids, 'num_tokens': self.L}


class PackedDataModule(pl.LightningDataModule):
    """Trains from pre-packed token files: zero tokenization at train time.
    Builds missing packs on first use. Datasets without validation/test
    splits get fractional 80/10/10 slices of the packed train stream (note:
    a different val partition than the streaming pipeline's, so val losses
    aren't comparable across the two data paths)."""

    def __init__(self, dataset_name, dataset_cfg, max_length=4096,
                 batch_size=8, num_workers=8):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        L = self.max_length
        train_p = prepack(self.dataset_name, self.dataset_cfg, 'train')
        try:
            val_p = prepack(self.dataset_name, self.dataset_cfg, 'validation')
            test_p = prepack(self.dataset_name, self.dataset_cfg, 'test')
            self.train_ds = PackedFileWindows(train_p, L)
            self.val_ds = PackedFileWindows(val_p, L)
            self.test_ds = PackedFileWindows(test_p, L)
        except ValueError:
            self.train_ds = PackedFileWindows(train_p, L, 0.0, 0.8)
            self.test_ds = PackedFileWindows(train_p, L, 0.8, 0.9)
            self.val_ds = PackedFileWindows(train_p, L, 0.9, 1.0)

    def _loader(self, ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=shuffle, persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.max_length = max_length
        self.tokenizer = _tokenizer()
        self.batch_size = batch_size
        self.streaming = False

    def data_loader(self, split):
        ds = load_dataset(self.dataset_name, self.dataset_cfg, split=split,
                          streaming=self.streaming)
        packed = PackedWindows(ds, self.max_length)
        # Every window is the same dense (max_length,) shape, so default
        # collation stacks them into (B, max_length). num_workers stays 0:
        # extra workers would each replay the same stream and duplicate data.
        return torch.utils.data.DataLoader(packed, batch_size=self.batch_size)

    def setup(self, stage=None):

        self.train_dataloader_ = self.data_loader('train')
        try:
            self.test_dataloader_ = self.data_loader('test')
            self.val_dataloader_ = self.data_loader('validation')
        except ValueError:
            try:
                self.train_dataloader_ = self.data_loader('train[0%:80%]')
                self.test_dataloader_ = self.data_loader('train[80%:90%]')
                self.val_dataloader_ = self.data_loader('train[90%:100%]')
            except ValueError:
                self.train_dataloader_ = self.data_loader('train')
                self.test_dataloader_ = self.data_loader('validation')
                self.val_dataloader_ = self.data_loader('validation')
        print("Done tokenizing.")

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_

    def test_dataloader(self):
        return self.test_dataloader_


class StreamingTextDataModule(BasicDataModule):
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8):
        super().__init__(dataset_name, dataset_cfg, max_length=max_length,
                         batch_size=batch_size)
        self.streaming = True

# `paths` must be imported before `datasets`/`transformers`: it's what points
# their caches at the shared datasettes mount, and they read the environment at
# import time.
import paths  # noqa: F401  (imported for its import-time side effects)

import torch
import datasets
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

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


def _source_columns(ds):
    """Columns to drop once we've tokenized: carrying raw text through the rest
    of the pipeline just burns memory and dataloader bandwidth. Streaming
    datasets don't always know their columns up front; None means "keep all"."""
    return list(ds.column_names) if ds.column_names else None


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


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8,
                 num_workers=20, min_tokens=8):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.max_length = max_length
        self.tokenizer = _tokenizer()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Line-oriented corpora (wikitext especially) are full of blank lines and
        # section headers that tokenize to a handful of tokens. They cost a full
        # batch slot and teach nothing, so drop them.
        self.min_tokens = min_tokens
        self.streaming = False

    def data_loader(self, split):
        cols = ['input_ids', 'num_tokens']

        def encode_truncated(s):
            t = encode(s,
                       add_special_tokens=True,
                       max_length=self.max_length,
                       truncation=True)
            return {'input_ids': t['input_ids']}

        def long_enough(s):
            return len(s['input_ids']) >= self.min_tokens

        def encode_truncated_ds(ds):
            assert not self.streaming
            ds = ds.map(encode_truncated, batched=True, num_proc=self.num_workers,
                        remove_columns=_source_columns(ds))
            ds = ds.filter(long_enough)
            ds.set_format(type="torch", columns=cols)
            return ds

        def collate_batch(batch):
            longest = 0
            out_batch = []
            num_tokens = []
            for l in batch['input_ids']:
                num_tokens += [len(l)]
                if len(l) > longest:
                    longest = len(l)
                out_batch.append(torch.tensor(l, dtype=torch.long))
            seq = pad_sequence(out_batch, batch_first=True, padding_value=pad_token_id())
            return {'input_ids': seq, 'num_tokens': num_tokens}

        def encode_ds_streaming(ds):
            # The collate map is doing something slightly sneaky: a batched map
            # emits len(value) rows, so returning a (B, T) tensor hands back B
            # rows that are already padded to a common length. Because streaming
            # preserves order, the DataLoader's batches line up exactly with
            # these map batches and re-assemble them into (B, T).
            return ds.map(encode_truncated, batched=True, batch_size=self.batch_size,
                          remove_columns=_source_columns(ds)). \
                filter(long_enough). \
                map(collate_batch, batched=True, batch_size=self.batch_size). \
                with_format(type="torch")

        def encode_ds(ds):
            if self.streaming:
                return encode_ds_streaming(ds)
            return encode_truncated_ds(ds)

        ds = load_dataset(self.dataset_name, self.dataset_cfg, split=split, streaming=self.streaming)
        ds = encode_ds(ds)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size)

    def setup(self, stage=None):
        if self.streaming:
            print("Streaming...")
            self.num_workers = 1
        else:
            print("Tokenizing...")

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
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8,
                 num_workers=20, min_tokens=8):
        super().__init__(dataset_name, dataset_cfg, max_length=max_length,
                         batch_size=batch_size, num_workers=num_workers,
                         min_tokens=min_tokens)
        self.streaming = True

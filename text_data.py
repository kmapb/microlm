import torch
import datasets
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertTokenizer, DataCollatorWithPadding
from torch.nn.utils.rnn import pad_sequence

TOKENIZER=None

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
    return len(_tokenizer().vocab)

def sep_token_id():
    return _tokenizer().sep_token_id

def encode(s, add_special_tokens=True, truncation=True, max_length=None):
    return _tokenizer()(s['text'],
                        add_special_tokens=add_special_tokens,
                        max_length=max_length,
                        truncation=truncation)

def decode(t):
    return _tokenizer().decode(t)

# e.g., dataset = load_dataset('the_pile', 'all', split='train', streaming=True)
def load_dataset(name, config, split='train', streaming=True, shuffle=True, num_proc=16):
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
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8, num_workers=20):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer = _tokenizer()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.streaming = False

    def data_loader(self, split):
        cols = ['input_ids']
        
        def encode_truncated(s):
            t = encode(s,
                       add_special_tokens=True,
                       max_length=self.max_length,
                       truncation=True)
            return { 'input_ids': t['input_ids'] }

        def encode_truncated_ds(ds):
            assert not self.streaming
            ds = ds.map(encode_truncated, batched=True, num_proc=self.num_workers)
            ds.set_format(type="torch", columns=cols)
            return ds
        
        def collate_batch(batch):
            max = 0
            out_batch = []
            for l in batch['input_ids']:
                if len(l) > max:
                    max = len(l)
                out_batch.append(torch.tensor(l, dtype=torch.long))
            seq = pad_sequence(out_batch, batch_first=True, padding_value=_tokenizer().pad_token_id)
            return { 'input_ids': seq }
        
        def encode_ds_streaming(ds):
            return ds.map(encode_truncated, batched=True, batch_size=self.batch_size). \
                map(collate_batch, batched=True, batch_size=self.batch_size). \
                with_format(type="torch")

        def encode_ds(ds):
            if self.streaming:
                return encode_ds_streaming(ds)
            return encode_truncated_ds(ds)

        ds = load_dataset(self.dataset_name, self.dataset_cfg, split=split, streaming=self.streaming)
        ds = encode_ds(ds)
        # collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                           padding='longest', max_length=self.max_length)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=self.batch_size,
                                           # collate_fn=collator,
                                           # num_workers=self.num_workers)
        )
    
    def setup(self, stage=None):
        if self.streaming:
            self.num_workers = 1

        print("Tokenizing...")
        self.train_dataloader_ = self.data_loader('train')
        try:
            self.test_dataloader_ = self.data_loader('test')
            self.val_dataset = self.data_loader('validation')
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
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=8, num_workers=20):
        super().__init__(dataset_name, dataset_cfg, max_length=max_length, batch_size=batch_size, num_workers=num_workers)
        self.streaming = True

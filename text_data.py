import torch
import datasets
import pytorch_lightning as pl
from transformers import AutoTokenizer, BertTokenizer, DataCollatorForLanguageModeling, DataCollatorWithPadding

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

def embatch(encoded, max_batch_size=16):
    assert len(encoded.shape) == 1
    # Shorten batches if they're too long
    T = encoded.shape[0]
    B = min(max_batch_size, T)
    assert T >= B
    
    x = torch.zeros(B, T, dtype=torch.long)
    x = x + _tokenizer().pad_token_id
    for i in range(0, B):
        x[i, B-i-1:] = encoded[:T-B+i+1]
    return x

# e.g., dataset = load_dataset('the_pile', 'all', split='train', streaming=True)
def load_dataset(name, config, split='train', streaming=True, shuffle=True, num_proc=16):
    ds = datasets.load_dataset(name, config, split=split, streaming=streaming, num_proc=num_proc)
    shuf = ds
    if shuffle:
        if streaming:
            shuf = ds.shuffle(buffer_size=8192)
        else:
            shuf = ds.shuffle()
    return shuf
# Do encoding/batching stuff elsewhere
    def encode_example(item):
        item['encoded'] = encode(item['text'])
        item['batchened'] = embatch(item['encoded'])
        return item
    enc = shuf.map(encode_example, num_proc=num_proc)
    return enc

class TextDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_cfg, streaming=False, pct=None, batch_size=20, num_workers=16):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().save_hyperparameters()
        pct = 1.0 if pct is None else pct
        def split_name(base):
            # return base
            return "{}[:{}%]".format(base, int(pct * 100))
        def ds(split):
            return load_dataset(dataset_name, dataset_cfg, split=split_name(split), streaming=streaming)
        self.train_dataset = ds('train')
        self.test_dataset = ds('test')
        self.val_dataset = ds('validation')

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return torch.DataLoader(self.train_dataset)
    def test_dataloader(self):
        return torch.DataLoader(self.test_dataset)
    def val_dataloader(self):
        return torch.DataLoader(self.val_dataset)
    
    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if 'batchened' in batch:
            b = batch['batchened']
            if isinstance(b, list):
                # import pdb; pdb.set_trace()
                return torch.tensor(b, dtype=torch.long).to(device)
        else:
            # Hmm, what in blazes is this?
            import pdb; pdb.set_trace()
            assert False
    
def epoch_gen(idata, batch_size, example_length, max_samples=None):
    num_samples = 0
    while True:
      to_cat = []
      obtained = 0
      # We get up to example_length + 1 to make sure we have room for the y's.
      print("nSamp {} maxSamp {}".format(num_samples, max_samples))
      while obtained < batch_size * (example_length + 1):
          if max_samples is not None and num_samples > max_samples:
              return
          num_samples += 1
          batch = next(idata, False)
          if batch is False:
              return
          enc = batch['encoded']
          obtained += enc.shape[0]
          to_cat.append(enc)
      xy_overlay = torch.cat(to_cat).split(batch_size * (example_length + 1))[0]
      xy_overlay = xy_overlay.view( (batch_size, example_length + 1))
      x = xy_overlay[:, 0:example_length]
      y = xy_overlay[:, -1]
      # Free up some memory
      enc = None
      to_cat = []
      xy_overlay = None
      # yield 'em
      yield x, y

def complete_prefix(m, init_str='Zounds! ', max_new_tokens=1024):
  init_context = encode(init_str, add_special_tokens=True) # Get [CLS]...[SEP]
  init_context = init_context[None, :-1] # Remove [SEP], add a batch
  return decode(m.generate(idx = init_context, max_new_tokens=max_new_tokens))

if __name__ == "__main__":
    ds = load_dataset("the_pile", config="pubmed", split='train', streaming=True)
    for x in ds.take(3):
        print(x['text'])
    it = iter(ds)
    for x,y in epoch_gen(it, 2, 10):
        print(x)
        print(y)
        break


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_cfg, max_length=4096, batch_size=128, num_workers=20):
        super().__init__()
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer = _tokenizer()
        self.batch_size = batch_size
        self.num_workers = num_workers
        def ds(split):
            return load_dataset(dataset_name, dataset_cfg, split=split, streaming=False)
        self.train_dataset = ds('train')
        try:
            self.test_dataset = ds('test')
            self.val_dataset = ds('validation')
        except ValueError:
            self.train_dataset = ds('train[0%:80%]')
            self.test_dataset = ds('train[80%:90%]')
            self.val_dataset = ds('train[90%:100%]')

    @staticmethod
    def collate_batch(batch):
        # import pdb; pdb.set_trace()
        from torch.nn.utils.rnn import pad_sequence
        max = 0
        out_batch = []
        for item in batch:
            t = item['input_ids']
            if t.shape[0] > max:
                max = t.shape[0]
            out_batch.append(t)
        out = pad_sequence(out_batch, batch_first=True, padding_value=max)
        return out

    def data_loader(self, ds):
        # collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        # XXX: make max_length paramterizable
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                           padding='longest', max_length=self.max_length)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=self.batch_size,
                                           collate_fn=collator,
                                           num_workers=self.num_workers)
    
    def setup(self, stage=None):
        cols = ['input_ids', 'token_type_ids', 'attention_mask']
        print("Tokenizing...")
        
        def encode_truncated(s):
            return encode(s, max_length=self.max_length, truncation=True)

        self.train_dataset = self.train_dataset.map(encode_truncated, batched=True, num_proc=self.num_workers)
        self.train_dataset.set_format(type="torch", columns=cols)
        self.train_dataloader_ = self.data_loader(self.train_dataset)
        
        self.val_dataset = self.val_dataset.map(encode_truncated, batched=True, num_proc=self.num_workers)
        self.val_dataset.set_format(type="torch", columns=cols)
        self.val_dataloader_ = self.data_loader(self.val_dataset)
        
        self.test_dataset = self.test_dataset.map(encode_truncated, batched=True, num_proc=self.num_workers)
        self.test_dataset.set_format(type="torch", columns=cols)
        self.test_dataloader_ = self.data_loader(self.test_dataset)
        
        print("Done tokenizing.")

    def train_dataloader(self):
        return self.train_dataloader_
        
    def val_dataloader(self):
        return self.val_dataloader_
 
    def test_dataloader(self):
        return self.test_dataloader_

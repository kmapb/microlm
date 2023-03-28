import torch
import datasets
import pytorch_lightning as pl
from transformers import BertTokenizer

TOKENIZER=None
def _tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
            TOKENIZER = BertTokenizer.from_pretrained("bert-base-cased")
    return TOKENIZER

def tokenize(text):
    return _tokenizer()(text)['input_ids']

def vocabulary_size():
    return len(_tokenizer().vocab)

def encode(s, add_special_tokens=True):
  return torch.tensor(
      _tokenizer()(s, add_special_tokens=add_special_tokens)['input_ids'],
      dtype=torch.long)

def decode(t):
    return _tokenizer().decode(t)

def embatch(encoded, max_batch_size=19):
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
def load_dataset(name, config, split='train', streaming=True, shuffle=True):
    ds = datasets.load_dataset(name, config, split=split, streaming=streaming)
    shuf = ds
    if shuffle:
        if streaming:
            shuf = ds.shuffle(buffer_size=8192)
        else:
            shuf = ds.shuffle()
    def encode_example(item):
        item['encoded'] = encode(item['text'])
        item['batchened'] = embatch(item['encoded'])
        return item
    enc = shuf.map(encode_example)
    return enc
    #  .map(embatchen)

class TextDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, dataset_cfg, streaming=False, batch_size=20, num_workers=20):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        super().save_hyperparameters()
        self.train_data_loader = load_dataset(dataset_name, dataset_cfg, split='train', streaming=streaming)
        self.test_data_loader = load_dataset(dataset_name, dataset_cfg, split='test', streaming=streaming)
        self.val_data_loader = load_dataset(dataset_name, dataset_cfg, split='validation', streaming=streaming)

    def setup(self, stage=None):
        pass
    
    def train_dataloader(self):
        return self.train_data_loader
    def test_dataloader(self):
        return self.test_data_loader
    def val_dataloader(self):
        return self.val_data_loader
    
    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        if 'batchened' in batch:
            b = batch['batchened']
            if isinstance(b, list):
                # import pdb; pdb.set_trace()
                return torch.tensor(b, dtype=torch.long).to(device)
        if False and 'encoded' in batch:
            t = batch['encoded']
            if isinstance(t, list):
                # Hmm, list batch? Concatenate it.
                t = torch.tensor(t, dtype=torch.long)
            # Unary batch. Stick a dummy batch dimension on it.
            return t.to(device)[None, :]
        else:
            # Hmm, what in blazes is this?
            import pdb; pdb.set_trace()
            pass
        return t
    
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
  init_context = encode(init_str, add_special_tokens=False)
  init_context = init_context[None, :]
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


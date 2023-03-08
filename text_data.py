import torch
from datasets import load_dataset
from transformers import BertTokenizer

def load_dataset(name, config, split='train', streaming=False):
    ds = load_dataset(name, cfg, split=split, streaming=True)
    if streaming:
        return ds.shuffle(buffer_size=8192)
    return ds.shuffle()

# e.g., dataset = load_dataset("the_pile", split='train', streaming=True)
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

def encode(s, max_length=8192, add_special_tokens=True):
  return torch.tensor(
      _tokenizer()(s, max_length=max_length, add_special_tokens=add_special_tokens)['input_ids'],
      dtype=torch.long)

decode = _tokenizer().decode





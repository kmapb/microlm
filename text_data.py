import torch
import datasets
from transformers import BertTokenizer

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

def load_dataset(name, config, split='train', streaming=True):
    ds = datasets.load_dataset(name, config, split=split, streaming=streaming)
    if streaming:
        shuf = ds.shuffle(buffer_size=8192)
    else:
        shuf = ds.shuffle()
    def encode_example(item):
        item['encoded'] = encode(item['text'])
        return item
    return shuf.map(encode_example)

if __name__ == "__main__":
    ds = load_dataset("the_pile", config="pubmed", split='train', streaming=True)
    for x in ds.take(3):
        print(x['text'])

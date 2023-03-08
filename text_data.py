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

def decode(t):
    return _tokenizer().decode(t)

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


def epoch_gen(idata, batch_size, example_length):
    while True:
      to_cat = []
      obtained = 0
      # We get up to example_length + 1 to make sure we have room for the y's.
      while obtained < batch_size * (example_length + 1):
          batch = next(idata, False)
          if batch is False:
              return
          enc = batch['encoded']
          obtained += enc.shape[0]
          to_cat.append(enc)
      xy_overlay = torch.cat(to_cat).split(batch_size * (example_length + 1))[0]
      xy_overlay = xy_overlay.view( (batch_size, example_length + 1))
      x = xy_overlay[:, 0:example_length]
      y = xy_overlay[:, 1:example_length+1]
      # Free up some memory
      enc = None
      to_cat = []
      xy_overlay = None
      # yield 'em
      yield x, y

def complete_prefix(m, init_str='Zounds! ', max_new_tokens=1024):
  init_context = encode(init_str, add_special_tokens=False)
  init_context = init_context[None, :]
  return decode(m.generate(idx = init_context, max_new_tokens=max_new_tokens)[0].tolist())

if __name__ == "__main__":
    ds = load_dataset("the_pile", config="pubmed", split='train', streaming=True)
    for x in ds.take(3):
        print(x['text'])
    it = iter(ds)
    for x,y in epoch_gen(it, 2, 10):
        print(x)
        print(y)
        break


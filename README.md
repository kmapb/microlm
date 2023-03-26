# microlm

A very smol language model. Basesd on tree-recurrence, so history requires
O(log n) recurrence steps for input length n. See `ReConvText` for details.

Training: `python train.py` will run for 48 hours or two epochs on a large
dataset. I've run it on wikitext-103, but other datasets seem like they
train up ok too. It drops a final model in `model.ckpt`.

Inference: `chat.py` loads `model.ckpt` and provides a text-completion UI.

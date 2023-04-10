# microlm

A very smol language model. Basesd on tree-recurrence, so history requires
O(log n) recurrence steps for input length n. See `SummNet` for details.

*Update:*  I've since learned that "WaveNet" is the way to talk about this
approach. Stacked, dilated convolutions, with 2^i dilation at layer i.

Training: `python train.py` will run for 48 hours or two epochs on a large
dataset. The default is wikitext-103, but other datasets seem like they
train up ok too. It drops a final model in `model.ckpt`.

Inference: `chat.py` loads `model.ckpt` and provides a text-completion UI.
Currently spews intermediate beam search and scores.

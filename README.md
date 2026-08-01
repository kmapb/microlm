# microlm

A very smol language model. Based on tree-recurrence, so history requires
O(log n) recurrence steps for input length n. See `SummNet` for details.

*Update:*  I've since learned that "WaveNet" is the way to talk about this
approach. Stacked, dilated convolutions, with 2^i dilation at layer i.

## Setup

```sh
python3 -m venv .venv
. .venv/bin/activate
# On a CPU-only machine, add the extra index so pip picks CPU torch wheels:
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# On a CUDA machine, plain `pip install -r requirements.txt` works.
```

## Training

`python train.py` streams `Salesforce/wikitext` (wikitext-103) by default,
trains for up to `--max-hours` (default 0.5) or `--max-epochs` (default 2),
and drops a final model in `full-run-d<dim>-h<kernel>.ckpt` (plus top-k
validation checkpoints under `val_ckpts/`). Other Huggingface text datasets
train up ok too via `--dataset`/`--dataset-cfg`; use the fully-namespaced
hub id (e.g. `Salesforce/wikitext`).

The accelerator is auto-detected, so it runs on CPU or GPU. Metrics go to a
local CSV logger under `logs/` by default; pass `--wandb` to log to Weights
& Biases instead (requires `pip install wandb`).

A quick CPU smoke run, small model on a small dataset:

```sh
python train.py --dataset Salesforce/wikitext --dataset-cfg wikitext-2-raw-v1 \
    --max-steps 30 --embedding-width 64 --fc-width 128 \
    --wavenet-height 6 --batch-size 2 --max-length 256
```

## Inference

`chat.py` loads a checkpoint and provides a text-completion UI:

```sh
python chat.py full-run-d64-h3.ckpt
```

## Tests

```sh
pytest
```

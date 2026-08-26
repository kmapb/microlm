# TODO: training-quality fixes

Five defects found in review (2026-08-26), roughly in order of expected impact
on model quality. Each one caps quality on its own; together they go a long way
toward explaining disappointing samples from otherwise-healthy training runs.

## 1. LeakyReLU on the output logits

`SummNet.head` ends `Linear(fc_dim, vocab) -> LeakyReLU()`, and that output goes
straight into `F.cross_entropy`. Cross-entropy expects unbounded logits;
LeakyReLU compresses all negative logits by 100x, so the model can never
confidently rule tokens *out* and the predictive distribution stays mushy no
matter how long it trains.

**Fix:** drop the trailing `LeakyReLU` from the head (the same bug exists in
`token_rnn.py` and `conv_text.py`). Note: removing it breaks checkpoint
compatibility in spirit if not in shape -- old checkpoints were trained to a
different objective geometry.

## 2. Loss is computed over padding

`collate_batch` pads ragged documents with `pad_token_id`, and the
`cross_entropy` call in `SummNet._shared_eval` has no `ignore_index`. Every
position after a document's `[SEP]` teaches the model "predict `[PAD]`", which
can be a large fraction of the gradient, and it deflates the logged loss.

**Fix:** `F.cross_entropy(..., ignore_index=pad_token_id)`; also log loss per
*real* token so train/val numbers are comparable across batches. Largely
subsumed by #4, but worth doing anyway for any padded path that remains.

## 3. Learning rate is ~30x too low, with no schedule

`configure_optimizers` returns AdamW at a constant `lr=1e-5` -- a fine-tuning
rate, not a from-scratch rate. No warmup, no decay, no gradient clipping.

**Fix:** peak lr ~3e-4 with linear warmup (~1-2k steps) and cosine decay;
`gradient_clip_val=1.0` on the Trainer. Expose peak lr and warmup as CLI args.

## 4. No sequence packing

Each dataset row is tokenized and padded individually, so line-oriented corpora
train the 4096-context WaveNet almost entirely on short sequences, and most of
each batch is padding. Compute goes to waste and long-range structure is never
learned.

**Fix:** concatenate-and-chunk -- tokenize the stream, join documents with
`[SEP]`, and emit fixed `max_length` windows with no padding at all. This also
eliminates the pad-loss issue (#2) on the packed path, and replaces the fragile
map/DataLoader batch-alignment trick in `text_data.py`.

## 5. Receptive field vastly exceeds the context

Dilations grow as `kernel_size ** h`; with the default k=3, height=11 the top
layers have dilation >> 4096, so they convolve almost pure left-padding.
Wasted parameters and compute.

**Fix:** cap the effective height so the receptive field just covers
`max_length` (for k=3, T=4096 that's height 8), either by validating the
height/kernel/length combination at construction or by deriving height from
`max_length` when unspecified.

## Worth doing while in there (not part of the five)

- WaveNet-style gated activations (GLU / tanh x sigmoid) instead of LeakyReLU
  inside the residual blocks, and skip-connection aggregation into the head.
- Tie the embedding and output projection weights.
- Prompt encoding in `chat.py` omits `[CLS]`, but training always prepends it:
  every prompt is out-of-distribution.

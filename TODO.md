# TODO: training-quality fixes

Five defects found in review (2026-08-26), roughly in order of expected impact
on model quality. **All five are now fixed** (see the `Fix N/5:` commits);
checkpoints from before that series were trained against a different objective
and aren't comparable. What each was:

1. **LeakyReLU on the output logits** -- the head fed cross-entropy activations
   with the negative half squashed 100x, so the model could never confidently
   rule a token out. Fixed: the final Linear's output is the logits, no
   activation (also in the legacy `token_rnn.py` / `conv_text.py`).
2. **Loss computed over padding** -- every position after a document's end
   taught "predict [PAD]". Fixed: `pad_token_id` is a SummNet hparam used as
   `ignore_index`, making the loss a per-real-token mean. (Mostly moot now
   that batches are packed, but it guards any padded path.)
3. **lr=1e-5, constant** -- a fine-tuning rate for a from-scratch model.
   Fixed: peak `--lr` 3e-4, linear `--warmup-steps`, cosine decay to a 10%
   floor over `--lr-decay-steps`, `gradient_clip_val=1.0`.
4. **No sequence packing** -- rows tokenized/padded individually, so training
   was mostly short contexts and pad slots. Fixed: `PackedWindows` joins
   documents with `[SEP]` and emits dense fixed-`max_length` windows; zero
   padding, and no more `[CLS]` train/inference mismatch with `chat.py`.
5. **Receptive field vastly exceeding the context** -- k=3/height=11 put the
   top layers' dilations past 4096; they convolved only left-padding. Fixed:
   height defaults to the smallest stack covering `max_length` (8 for the
   defaults); explicit overshoots warn.

## Still worth doing

- `chat.py`'s HF GenerationMixin shim no longer survives modern transformers
  (cache prep wants a real model config: `num_hidden_layers` etc.). A plain
  top-k/top-p sampling loop over `SummNet.forward` is ~15 lines and drops the
  transformers surface entirely.
- Log a few fixed-prompt samples at each validation pass (to MLflow) so
  subjective quality is visible per checkpoint without manual sampling.

- WaveNet-style gated activations (GLU / tanh x sigmoid) instead of LeakyReLU
  inside the residual blocks, and skip-connection aggregation into the head.
- Tie the embedding and output projection weights.
- A real long run to re-baseline: wikitext-103, compare against GCNN-8 (44.9
  test PPL) / TCN-class numbers. Note `test_loss` is now nats/token over
  packed windows; ppl = exp(loss).
- Tokenizing in the training process is the only data-path worker now; if the
  GPU starves on a big run, move `PackedWindows` behind a worker or pre-pack
  to disk.

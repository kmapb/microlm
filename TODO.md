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

## Proposed: gated blocks + skip aggregation (arch v2)

**Done, and measured** (2026-08-28). Controlled ladder on fineweb-edu, ~90M
params, 1.8B tokens, identical recipe -- final val / fineweb test / wikitext
zero-shot:

    v1 (baseline convs)   4.073 / 4.082 / 5.007
    v2 (+GLU, skip, tie)  3.895 / 3.903 / 4.812
    v3 (v2 minus PE)      3.786 / 3.793 / 4.693   <- reference conv arch
    t1 (GPT-2-ish, 9x768) 3.365 / 3.379 / 4.224   <- attention premium: 0.42 nats

v3 is the conv architecture going forward; the 0.42-nat gap to t1 at equal
params/data is the target for dilated attention (v4). Original design notes
(`arch='v2'` hparam; absent-key default `'v1'` keeps old checkpoints
loading):

1. **GLU gating** (Dauphin et al. 2017). Replace each block's
   `conv -> leaky_relu` with a single dilated causal conv to `2*dim` channels,
   split into value/gate halves: `v, g = conv(x).chunk(2, dim=1); h = v *
   sigmoid(g)`. The linear (untanh'd) value path is GCNN's headline result
   over WaveNet's tanh x sigmoid -- keeps a linear gradient path through depth.
2. **Pre-norm residual blocks.** `x + f(LayerNorm(x))` instead of the current
   post-norm `LayerNorm(x + f(x))`; the residual stream stays un-normalized,
   which trains more stably and composes with skip taps.
3. **Skip aggregation** (WaveNet). Each block also emits `skip_l = W_l h_l`
   (per-layer 1x1 conv); the head consumes `LayerNorm(sum_l skip_l)` instead
   of the top of the stack. Gives the loss a path-length-1 gradient to every
   depth and lets shallow layers (n-gram features) feed the prediction
   directly.

Plus **weight tying** to pay for it: tie the head's final `Linear(fc_dim,
vocab)` to the embedding table (works because fc_dim == dim == 1024). Param
arithmetic at d=1024/h=8: GLU doubles conv width (+25M), skip 1x1s +8M,
tying -30M -> ~93M total, within 4% of v1's 89.8M, so the fineweb comparison
stays fair at the same 1.8B-token budget.

Controlled experiment: identical recipe (3e-4, warmup 1k, cosine to floor at
55k steps, fineweb-edu sample-10BT, single pass), same MLflow experiment;
readout is val/test delta vs fineweb-edu-k3-d1024's 4.073/4.082.

## Scaling to ~1B (planned 2026-08-27)

Two work items that gate a ~1B-param run, in order:

1. **Data-path throughput first.** Current runs sit at ~20-50% GPU util
   because a single in-process tokenizer feeds the A100. Before any longer
   run, fix the input pipeline: either (a) pre-tokenize once to a
   memory-mapped uint16 token file on the datasettes mount (~2 bytes/token,
   so 20B tokens = 40GB; training then reads dense windows with zero
   tokenization cost -- simplest and probably best), or (b) proper
   multi-worker loading with per-worker stream shards. Target: >90% util.
   Assume this lands before sizing the big run's wall-clock.

2. **Hyperparameter autoresearch for the ~1B parameterization.** Questions
   to settle with a small scaling ladder (e.g. 94M -> 250M -> 1B) rather
   than vibes: deeper vs wider vs both; whether to adopt *repeated dilation
   cycles* (WaveNet-style: dilations 1..k^8 repeated x2-3) so depth can grow
   without the receptive field exploding past useful context; context length
   (8k? 16k?); LR scaling with width (muP-style transfer or an empirical
   sweep at 250M); batch size / tokens-per-step at 1B. Starting point for
   the arithmetic at k=3, GLU, tied embeddings: dim 3072 x 12 blocks is
   ~0.9B params; dim 2048 x (9-block cycle x 3) is ~1.0B with depth 27.

Dataset for the 1B run: Chinchilla wants >=20B tokens. Recommended:
`HuggingFaceFW/fineweb-edu` config `sample-100BT` -- same distribution as the
current runs (clean continuity for cross-scale comparisons), 100B tokens so a
single-epoch 20-30B subset never repeats. Stronger-mix alternative if we're
willing to break continuity: `mlfoundations/dclm-baseline-1.0`.

## Planned: v4 -- QKV spanning tree (dilated attention)

Design agreed 2026-08-28. Target: close part of the measured 0.42-nat gap
between v3 (3.786) and t1 (3.365) while staying O(T log T) and
length-agnostic. Lives side by side with v3: new module `tree_attention.py`,
`arch='v4'`; the v3 classes are not touched.

**Core op (`DilatedAttention`)**: at a layer with dilation d and window w,
position i attends over the candidate set {i - j*d : j = 0..w-1} -- multi-head
q.k softmax over w slots, plus a learned per-slot, per-head bias (T5-style,
w x heads params). The bias is the exact analogue of the conv kernel's
per-offset weights, so uniform-query behavior recovers a dilated conv as a
special case; content-dependence is strictly added expressivity. No global
PE (v3 lesson); slot bias carries local offset identity. RoPE on q/k is the
ablation alternative if slot bias underperforms.

**Implementation**: per slot j, left-pad K/V by j*d in time and slice (a
w-iteration python loop of tensor slices, w is small); scores (B, H, T, w),
mask slots reaching before t=0, softmax over w, weighted-sum values, output
projection. Memory ~w x the K/V tensors -- fine at w=8, T=4096, A100.

**Block/stack**: reuse the v3 macro-structure unchanged -- pre-norm block,
residual stream, per-layer 1x1 skip tap, head consumes LayerNorm(sum of
skips), tied embeddings with 0.02 init. The ONLY change vs v3 is
GatedCausalConv1d -> DilatedAttention, keeping the ablation clean.

**Schedule**: dilations d = w^l for l in 0..L-1 (one cycle covers w^L >=
max_length; w=8, T=4096 -> L=4), repeated for C cycles (default 3 -> 12
layers, ~94M params at dim 1024 -- in family with v2/v3/t1). New hparams
`window` (8), `cycles` (3); CLI --window/--cycles.

**Validation before the 5h run**: (1) unit tests -- causality parametrized,
full-context reachability (perturb token 0, prediction at T-1 moves), tying/
trains; (2) a 10-minute synthetic associative-recall probe (Zoology-style:
key-value pairs in context, query at the end) at tiny scale comparing
v3/v4/t1 -- v4's reason to exist is that probe, so measure it first.

**Run**: identical fineweb recipe (55k steps, 1.8B tokens), run name
fineweb-edu-v4-*; readout vs v3 3.786 / t1 3.365, plus wikitext cross-eval
and samples.

**Known risks**: value must survive C*L relay hops for exact long-range
recall (cycles >= 2 so late cycles re-route with better keys); slot softmax
includes j=0 self, giving a natural pass-through path; watch GPU util on the
slice-loop implementation.
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

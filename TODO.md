# microlm: results ledger and work queue

Reorganized 2026-09-07 after three weeks of appends. History: the repo's
original five training defects (logit LeakyReLU, pad-loss, lr=1e-5, no
packing, oversized receptive field) were found and fixed 2026-08-26/27 --
see the `Fix N/5:` commits. Everything below postdates those fixes.

## Results ledger

**4k-context architecture ladder** (fineweb-edu sample-10BT, 1.8B tokens,
32,768 tokens/step, lr 3e-4; val / fineweb test / wikitext-103 zero-shot):

    v1  baseline convs        4.073 / 4.082 / 5.007
    v2  +GLU, skip, tie       3.895 / 3.903 / 4.812
    v3  v2 minus PE           3.786 / 3.793 / 4.693   reference conv arch
    v4  QKV tree w=8  (4 hop) 3.673 / 3.677 / 4.481
    v4  w=16 (3 hops)         3.595 / 3.605 /   --
    v4  w=64 (2 hops)         3.546 / 3.557 / 4.307
    v4  w=256 (2 hops)        3.521 / 3.534 /   --    base-tree champion
    v4  w=4096 (flat, NoPE)   3.659 / 3.673 /   --    trend inverts (no pos.)
    v4m w=64 +MLPs (d768)     3.423 / 3.434 /   --    ~86% of premium, -6% params
    t1  transformer (9x768)   3.365 / 3.379 / 4.224   the attention premium: 0.42

Decomposition of the 0.42-nat conv->transformer premium: content routing
~27%, fan-out (4->2 hops) ~+30%, MLP capacity ~+29%, residual ~14%.
Side findings: NoPE beats learned PE for all tree/conv archs; free-running
repetition loops strengthen with window size (copy circuits) and are
tamed by MLPs + sampler penalties; the w=4096 point is confounded (no
positional signal at all) -- treat as an attn-only NoPE transformer.

**16k context** (equal 1.8B tokens, 2x16384/step):

    v4m w=256   val 3.460 / test 3.475   84.3M    175M FLOPs/tok   143k tok/s
    t1          val 3.439 / test 3.455   99.3M    400M FLOPs/tok   101k tok/s

Gap shrank 0.058 -> ~0.020 nats (noise territory) at 44% of the FLOPs and
1.42x speed. Both lose loss 4k->16k: fineweb docs are short (5% of tokens
in docs >16k), so long context is mostly cost here.

**64k context, PG19** (equal 1.8B tokens, 1x65536/step, one epoch) -- the
first true long-document capability test, and the tree's first outright win:

    v4m w=256   val 3.578 / test 3.459   157k tok/s    3.0h    175M F/tok
    t1          val 4.118 / test 4.019    42k tok/s   11.8h   1079M F/tok

+0.54 nats at 3.9x speed and ~1/6 FLOPs; t1's slow start (65k-position
attention + 50M-param PE table on 1 seq/step) never recovers in-budget.
Caveats: iso-compute would let t1 close much of the gap (which is the
tree's argument, restated); fair modern baseline would use RoPE, not
GPT-2 learned PE. Subjectively v4m writes coherent Victorian prose with
correct dialogue attribution and abbreviation-expansion coreference
("the Rev." -> "The Reverend Mr. Jordan"); t1 derails ("cried the door").

**Batch/LR sweep** (2026-09-07; v4m-w256 @4k, loss at equal 650M tokens):

    32k tok/step (b8):  lr 3e-4 -> 3.630   lr 6e-4 -> 3.545
    65k tok/step (b16): lr 3e-4 -> 3.736   6e-4 -> 3.617   1.2e-3 -> 3.556
    131k (b32): OOM in backward at this model size on one A100.

Verdicts: (1) the historical recipe was under-LR'd -- 6e-4 at the standard
batch is worth ~0.085 nats at 650M tokens (uniform across the ladder, so
past comparisons stand, but future runs should use 6e-4); (2) the linear-
scaling diagonal (b8@6e-4 ~= b16@1.2e-3) says we are at/below critical
batch size through 65k tokens/step -- the 1B run can go wide; (3) loss was
still improving with LR at both batches, so the true optimum may be
slightly higher; probe 1.2e-3@b8 before locking the 1B recipe.

## Dialed recipe (for the 1B ladder)

arch v4m, window 256, cycles per context (2 hops), d=768+ scaled, NoPE,
tied embeddings w/ 0.02 init, packed data path (98% GPU util), lr 6e-4
(pending one more probe), warmup ~33M tokens, cosine to 10% floor at
budget end, grad clip 1.0, tokens/step up to 65k per A100.

## Work queue

1. **~1B scaling ladder** (94M -> 250M -> 1B). Shapes to test: wider vs
   deeper at fixed w=256/2-hop; d=2048-3072. Data: fineweb-edu
   sample-100BT (pack it; ~20-30B tokens/run). LR transfer: verify the
   sweep's LR holds at 250M before committing 1B GPU-time.
2. **RoPE t1 baseline** for any write-up of the long-context result.
3. **Repetition metric** (distinct-n per checkpoint) so "loops more" is a
   number; sampler already has min-p + rep-penalty (scripts/sample.py).
4. **Associative-recall probe** (scripts/recall_probe.py) is still not a
   trustworthy discriminator (all archs plateau ~0.38 at probe scale);
   needs Zoology-reference config before drawing capability conclusions.
5. **chat.py rewrite**: HF GenerationMixin shim is dead under modern
   transformers; fold scripts/sample.py logic in.
6. **MLflow backfills pending** (server had 503 outage 09-03+): t1-16k,
   both PG19-64k runs, and the sweep ran on tensorboard logging --
   backfill via scripts/backfill_mlflow.py when the server returns.
   Ops: mlflow.pbd.vc (Northflank) has had two incidents; train.py's
   FailSoftMLFlowLogger now survives both mid-run blips and setup-time
   outages.

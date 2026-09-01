"""Analytic forward-FLOPs-per-token estimates for the microlm archs.

Multiply-accumulate counted as 2 FLOPs; norms/softmax/embedding lookups
ignored (sub-1% at these shapes). t1's attention term uses the causal
average context T/2 -- the term that makes full attention's cost grow
with context length while the tree families stay flat.

These are *estimates for accounting*, logged alongside measured
throughput so runs can be compared on loss-vs-FLOPs and loss-vs-seconds
axes; they are not a substitute for a profiler.
"""


def fwd_flops_per_token(hp) -> float:
    """hp: a SummNet hparams namespace (arch, dim, fc_dim, height, ...)."""
    d, V = hp.dim, hp.vocab_size
    head = 2 * d * hp.fc_dim + 2 * hp.fc_dim * V
    arch = hp.arch

    if arch in ('v1', 'v2', 'v3'):
        # Dilated conv d -> 2d (GLU, v2/v3) or d -> d (v1), + 1x1 skip (v2/v3).
        conv = 2 * hp.kernel_size * d * (2 * d if arch != 'v1' else d)
        skip = 2 * d * d if arch != 'v1' else 0
        block = conv + skip
    elif arch in ('v4', 'v4m'):
        block = 8 * d * d            # qkv + out proj
        block += 4 * hp.window * d   # scores + weighted values
        block += 2 * d * d           # skip tap
        if arch == 'v4m':
            block += 16 * d * d      # 4x GELU MLP
    elif arch == 't1':
        block = 8 * d * d            # qkv + out proj
        block += 4 * (hp.max_length / 2) * d   # causal-average attention
        block += 16 * d * d          # 4x GELU MLP
    else:
        raise ValueError(f"unknown arch {arch!r}")

    return float(hp.height * block + head)

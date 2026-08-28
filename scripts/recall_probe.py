"""Synthetic associative-recall probe (MQAR-style, dense supervision).

Sequences are [k1 v1 ... kn vn | kq1 vq1 ... kqn vqn] where the query
section replays every key in shuffled order; the model is supervised at
each query position to emit that key's value. This is the capability
dilated attention (v4) exists to add over fixed conv kernels (v3), with
full attention (t1) as the reference.

STATUS (2026-08-28): not yet a trustworthy discriminator. At dim 128 /
3000 steps, v3, v4, and t1 all plateau at ~0.38 accuracy (chance 0.03)
across a 3e-4..3e-3 lr sweep -- a shared ceiling, insensitive to
architecture, meaning some common harness factor (training length, the
shared tied-embedding head, scale) binds before routing ability does.
Zoology-style separations likely need longer training and a config
closer to their reference setups. Findings so far should not be read as
"v4 doesn't help recall".

    uv run python scripts/recall_probe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from summ_net import SummNet
from util import dev

N_KEYS = 32
VOCAB = 2 * N_KEYS + 1          # keys 1..32, values 33..64, 0 unused


def make_batch(batch, pairs, generator):
    """[pair section | shuffled query section]; returns (inputs, query
    positions, value targets)."""
    B = batch
    keys = torch.stack([torch.randperm(N_KEYS, generator=generator)[:pairs] + 1
                        for _ in range(B)])
    vals = torch.randint(N_KEYS + 1, 2 * N_KEYS + 1, (B, pairs),
                         generator=generator)
    pair_sec = torch.stack([keys, vals], dim=2).reshape(B, 2 * pairs)
    perm = torch.stack([torch.randperm(pairs, generator=generator)
                        for _ in range(B)])
    kq, vq = keys.gather(1, perm), vals.gather(1, perm)
    qsec = torch.stack([kq, vq], dim=2).reshape(B, 2 * pairs)
    qpos = 2 * pairs + 2 * torch.arange(pairs)
    return torch.cat([pair_sec, qsec], dim=1), qpos, vq


def run(arch, pairs=8, steps=3000, batch=128, lr=1e-3, dim=128, seed=7):
    T = 4 * pairs
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    kwargs = dict(vocab_size=VOCAB, dim=dim, fc_dim=dim, max_length=T,
                  kernel_size=3, window=4, cycles=2)
    if arch == 't1':
        kwargs['height'] = 4
    model = SummNet(arch=arch, **kwargs).to(dev())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        x, qpos, y = make_batch(batch, pairs, g)
        x, y = x.to(dev()), y.to(dev())
        logits = model(x).reshape(batch, T, VOCAB)[:, qpos]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    hits = total = 0
    with torch.no_grad():
        for _ in range(16):
            x, qpos, y = make_batch(batch, pairs, g)
            x, y = x.to(dev()), y.to(dev())
            pred = model(x).reshape(batch, T, VOCAB)[:, qpos].argmax(-1)
            hits += int((pred == y).sum())
            total += y.numel()
    return hits / total


if __name__ == '__main__':
    print(f"MQAR accuracy (chance ~= {1.0 / N_KEYS:.3f}):")
    for arch in ('v3', 'v4', 't1'):
        accs = {lr: run(arch, lr=lr) for lr in (3e-4, 1e-3, 3e-3)}
        print(f"  {arch}: " +
              "  ".join(f"lr={lr:g}:{a:.3f}" for lr, a in accs.items()))

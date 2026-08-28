"""Dilated QKV attention on the WaveNet routing topology (arch 'v4').

Same spanning tree as the conv stacks -- a layer at dilation d connects
position i to {i - j*d : j = 0..window-1} -- but the edge weights are
content-computed q.k attention over that candidate set instead of fixed
kernel weights. O(T * window * log T) like the convs, unlike full
attention's O(T^2); every hop becomes a selective relay.

Kept separate from summ_net's conv classes on purpose: v3 and v4 are
side-by-side architectures, not revisions of one another.
"""

import math

import torch
from torch import nn
from torch import Tensor as Tens
from torch.nn import functional as F


class DilatedAttention(nn.Module):
    """Multi-head attention restricted to the dilated candidate set.

    The learned per-slot, per-head bias is the analogue of a conv kernel's
    per-offset weights: with uninformative queries the layer degrades to
    (roughly) a dilated conv, so content-routing is strictly added
    expressivity over the v3 blocks.
    """

    def __init__(self, channels: int, window: int, dilation: int):
        super(DilatedAttention, self).__init__()
        self.window = window
        self.dilation = dilation
        self.n_heads = max(1, channels // 64)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.slot_bias = nn.Parameter(torch.zeros(self.n_heads, window))

    def forward(self, x: Tens):
        """(B, T, C) -> (B, T, C)."""
        B, T, C = x.shape
        H, w, d = self.n_heads, self.window, self.dilation
        hd = C // H

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)              # (B, H, T, hd)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        # Slot j holds the key/value from j*d steps back: left-pad in time,
        # then slice. window is small, so a python loop of slices is fine.
        ks, vs = [], []
        for j in range(w):
            shift = j * d
            if shift == 0:
                ks.append(k)
                vs.append(v)
            else:
                ks.append(F.pad(k, (0, 0, shift, 0))[:, :, :T])
                vs.append(F.pad(v, (0, 0, shift, 0))[:, :, :T])
        K = torch.stack(ks, dim=3)                           # (B, H, T, w, hd)
        V = torch.stack(vs, dim=3)

        scores = torch.einsum('bhtc,bhtjc->bhtj', q, K) / math.sqrt(hd)
        scores = scores + self.slot_bias.view(1, H, 1, w)
        # Slot j is invalid where i - j*d reaches before the sequence start.
        # Slot 0 (self) is always valid, so the softmax never sees all -inf.
        t_idx = torch.arange(T, device=x.device).view(T, 1)
        j_idx = torch.arange(w, device=x.device).view(1, w)
        invalid = (t_idx - j_idx * d) < 0
        scores = scores.masked_fill(invalid.view(1, 1, T, w), float('-inf'))

        out = torch.einsum('bhtj,bhtjc->bhtc', torch.softmax(scores, dim=-1), V)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TreeBlock(nn.Module):
    """Pre-norm residual block with a skip tap -- the v3 GatedBlock shape,
    with DilatedAttention in place of the gated conv."""

    def __init__(self, channels: int, window: int, dilation: int):
        super(TreeBlock, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = DilatedAttention(channels, window, dilation)
        self.skip = nn.Linear(channels, channels)

    def forward(self, x: Tens):
        h = self.attn(self.norm(x))
        return x + h, self.skip(h)


class TreeAttentionNet(nn.Module):
    """C cycles of the dilation ladder w^0..w^(levels-1); output is the
    normalized sum of every block's skip tap, as in the v2/v3 stacks.
    External interface is (B, C, T) to match the other filter banks."""

    def __init__(self, channels: int, window: int, levels: int, cycles: int):
        super(TreeAttentionNet, self).__init__()
        self.blocks = nn.ModuleList(
            [TreeBlock(channels, window, dilation=window ** l)
             for _ in range(cycles) for l in range(levels)])
        self.out_norm = nn.LayerNorm(channels)
        self.window, self.levels, self.cycles = window, levels, cycles
        self.height = len(self.blocks)

    def forward(self, x: Tens):
        y = x.permute(0, 2, 1)
        total = torch.zeros_like(y)
        for block in self.blocks:
            y, skip = block(y)
            total = total + skip
        return self.out_norm(total).permute(0, 2, 1)

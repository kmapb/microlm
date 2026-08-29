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

    def _bias(self, rel: Tens):
        """Per-head additive bias for backward offsets `rel` (in units of the
        dilation); -inf where rel is outside [0, window)."""
        H, w = self.n_heads, self.window
        valid = (rel >= 0) & (rel < w)
        b = self.slot_bias[:, rel.clamp(0, w - 1)]           # (H, *rel.shape)
        return torch.where(valid.unsqueeze(0), b,
                           torch.full_like(b, float('-inf')))

    def forward(self, x: Tens):
        """(B, T, C) -> (B, T, C).

        Fold trick: attending over {i - j*d} is ordinary sliding-window
        causal attention on the d interleaved residue-class subsequences.
        Folding d into the batch turns the whole op into dense SDPA calls
        (flash/efficient kernels) -- no per-slot loops, no giant stacks.
        """
        B, T, C = x.shape
        H, w, d = self.n_heads, self.window, self.dilation
        hd = C // H

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)              # (B, H, T, hd)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        # Right-pad time to a multiple of d, then fold residue classes into
        # the batch: (B, H, T, hd) -> (B, H, d, Tf, hd). Padded queries are
        # sliced off at the end; padded keys sit in the future of every real
        # query, so causality already hides them.
        Tp = -(-T // d) * d
        if Tp != T:
            q, k, v = (F.pad(t, (0, 0, 0, Tp - T)) for t in (q, k, v))
        Tf = Tp // d
        q, k, v = (t.view(B, H, Tf, d, hd).permute(0, 1, 3, 2, 4)
                   for t in (q, k, v))

        idx = torch.arange(Tf, device=x.device)
        if Tf <= w:
            # One SDPA over the whole folded sequence; rel = i - j.
            mask = self._bias(idx.view(Tf, 1) - idx.view(1, Tf))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask.view(1, H, 1, Tf, Tf).to(q.dtype))
        else:
            # Chunked local attention: pad Tf to n*w chunks; each chunk
            # attends to [previous chunk | own chunk] under the band mask.
            Tc = -(-Tf // w) * w
            if Tc != Tf:
                q, k, v = (F.pad(t, (0, 0, 0, Tc - Tf)) for t in (q, k, v))
            n = Tc // w
            qc = q.view(B, H, d, n, w, hd)
            kc = k.view(B, H, d, n, w, hd)
            vc = v.view(B, H, d, n, w, hd)

            li = torch.arange(w, device=x.device)
            # Chunk 0: own chunk only; rel = i - j.
            m0 = self._bias(li.view(w, 1) - li.view(1, w))
            o0 = F.scaled_dot_product_attention(
                qc[:, :, :, 0], kc[:, :, :, 0], vc[:, :, :, 0],
                attn_mask=m0.view(1, H, 1, w, w).to(q.dtype))
            outs = [o0.unsqueeze(3)]
            if n > 1:
                kcat = torch.cat([kc[:, :, :, :-1], kc[:, :, :, 1:]], dim=4)
                vcat = torch.cat([vc[:, :, :, :-1], vc[:, :, :, 1:]], dim=4)
                # Key column j spans prev chunk (j < w) and own chunk;
                # rel = (i + w) - j.
                mr = self._bias(li.view(w, 1) + w - torch.arange(2 * w, device=x.device).view(1, 2 * w))
                orest = F.scaled_dot_product_attention(
                    qc[:, :, :, 1:], kcat, vcat,
                    attn_mask=mr.view(1, H, 1, 1, w, 2 * w).to(q.dtype))
                outs.append(orest)
            out = torch.cat(outs, dim=3).reshape(B, H, d, Tc, hd)[:, :, :, :Tf]

        # Unfold: (B, H, d, Tf, hd) -> (B, H, T, hd) -> (B, T, C).
        out = out.permute(0, 1, 3, 2, 4).reshape(B, H, Tp, hd)[:, :, :T]
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

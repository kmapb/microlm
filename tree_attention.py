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

# Above this folded length, a single-hop level skips the learned slot bias
# (see forward): building the (Tf, Tf) bias dominates runtime, and at that
# scale it's a relative PE this family measurably doesn't want (v3 result).
BIAS_MAX_LEN = 512


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
        # the *batch* dim: (B, H, T, hd) -> (B*d, H, Tf, hd). Keeping inputs
        # 4-D matters: 5-D knocks SDPA off its fused kernels onto the math
        # backend, which materializes fp32 (T, T) scores and OOMs at large
        # windows. Padded queries are sliced off at the end; padded keys sit
        # in the future of every real query, so causality hides them.
        Tp = -(-T // d) * d
        if Tp != T:
            q, k, v = (F.pad(t, (0, 0, 0, Tp - T)) for t in (q, k, v))
        Tf = Tp // d
        # t = f*d + r: split time into (fold f, residue r), move r to batch.
        q, k, v = (t.view(B, H, Tf, d, hd).permute(0, 3, 1, 2, 4)
                    .reshape(B * d, H, Tf, hd) for t in (q, k, v))

        idx = torch.arange(Tf, device=x.device)
        if Tf <= w:
            if Tf > BIAS_MAX_LEN:
                # Full-context hop: a learned (Tf, Tf) bias costs more to
                # build (and backprop) than the attention itself, and at this
                # scale it's just a relative PE -- which this family has
                # measured it doesn't want (v3). Pure NoPE causal flash;
                # slot_bias goes unused at this level.
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # One SDPA over the whole folded sequence; rel = i - j.
                mask = self._bias(idx.view(Tf, 1) - idx.view(1, Tf))
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask.view(1, H, Tf, Tf).to(q.dtype))
        else:
            # Chunked local attention: pad Tf to n*w chunks; each chunk
            # attends to [previous chunk | own chunk] under the band mask.
            Tc = -(-Tf // w) * w
            if Tc != Tf:
                q, k, v = (F.pad(t, (0, 0, 0, Tc - Tf)) for t in (q, k, v))
            n = Tc // w
            qc = q.view(B * d, H, n, w, hd)
            kc = k.view(B * d, H, n, w, hd)
            vc = v.view(B * d, H, n, w, hd)

            li = torch.arange(w, device=x.device)
            # Chunk 0: own chunk only; rel = i - j.
            m0 = self._bias(li.view(w, 1) - li.view(1, w))
            o0 = F.scaled_dot_product_attention(
                qc[:, :, 0], kc[:, :, 0], vc[:, :, 0],
                attn_mask=m0.view(1, H, w, w).to(q.dtype))
            outs = [o0.unsqueeze(2)]
            if n > 1:
                def merge(t):  # (B*d, H, n-1, L, hd) -> (B*d*(n-1), H, L, hd)
                    return t.permute(0, 2, 1, 3, 4).reshape(-1, H, t.shape[3], hd)
                kcat = torch.cat([kc[:, :, :-1], kc[:, :, 1:]], dim=3)
                vcat = torch.cat([vc[:, :, :-1], vc[:, :, 1:]], dim=3)
                # Key column j spans prev chunk (j < w) and own chunk;
                # rel = (i + w) - j.
                mr = self._bias(li.view(w, 1) + w -
                                torch.arange(2 * w, device=x.device).view(1, 2 * w))
                orest = F.scaled_dot_product_attention(
                    merge(qc[:, :, 1:]), merge(kcat), merge(vcat),
                    attn_mask=mr.view(1, H, w, 2 * w).to(q.dtype))
                orest = orest.view(B * d, n - 1, H, w, hd).permute(0, 2, 1, 3, 4)
                outs.append(orest)
            out = torch.cat(outs, dim=2).reshape(B * d, H, Tc, hd)[:, :, :Tf]

        # Unfold: (B*d, H, Tf, hd) -> (B, H, Tp, hd) -> (B, T, C).
        out = (out.view(B, d, H, Tf, hd).permute(0, 2, 3, 1, 4)
                  .reshape(B, H, Tp, hd)[:, :, :T])
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TreeBlock(nn.Module):
    """Pre-norm residual block with a skip tap -- the v3 GatedBlock shape,
    with DilatedAttention in place of the gated conv. With mlp=True ('v4m')
    a transformer-style 4x GELU MLP sub-layer follows the attention; the
    skip tap then carries attention + MLP contributions, since the head
    only sees the skip sum and a last-block MLP would otherwise be dead."""

    def __init__(self, channels: int, window: int, dilation: int,
                 mlp: bool=False):
        super(TreeBlock, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = DilatedAttention(channels, window, dilation)
        self.skip = nn.Linear(channels, channels)
        if mlp:
            self.mlp_norm = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(nn.Linear(channels, 4 * channels),
                                     nn.GELU(),
                                     nn.Linear(4 * channels, channels))
        else:
            self.mlp = None

    def forward(self, x: Tens):
        h = self.attn(self.norm(x))
        y = x + h
        if self.mlp is not None:
            m = self.mlp(self.mlp_norm(y))
            return y + m, self.skip(h + m)
        return y, self.skip(h)


class TreeAttentionNet(nn.Module):
    """C cycles of the dilation ladder w^0..w^(levels-1); output is the
    normalized sum of every block's skip tap, as in the v2/v3 stacks.
    External interface is (B, C, T) to match the other filter banks."""

    def __init__(self, channels: int, window: int, levels: int, cycles: int,
                 mlp: bool=False):
        super(TreeAttentionNet, self).__init__()
        self.blocks = nn.ModuleList(
            [TreeBlock(channels, window, dilation=window ** l, mlp=mlp)
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

import math
import warnings

import torch
from torch import nn
from torch import Tensor as Tens
from torch.nn import functional as F
import pytorch_lightning as pl
import util
import flops
from tree_attention import TreeAttentionNet
from typing import cast, Dict, Optional
import datetime as dt

__CUDA__ = torch.cuda.is_available()

def conv1d_factory(kernel_size: int, in_channels: int, out_channels: int, dilation: int=1):
    return CausalConv1d(kernel_size, in_channels, out_channels, dilation)

class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """
    def __init__(self, kernel_size: int, in_channels: int, out_channels: int, dilation: int=1):
        super(CausalConv1d, self).__init__()
        
        # attributes:
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        
        # modules:
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                      kernel_size, stride=1,
                                      padding=(kernel_size-1) * dilation,
                                      dilation=dilation)

    def forward(self, seq: Tens):
        """
        Expects a 3D tensor of shape (batch_size, channels, seq_len).
        """
        B,C,T = seq.shape
        assert(C == self.in_channels)
        #print("QQQ {}, dil {}".format(seq.shape, self.dilation))
        conv1d_out = self.conv1d(seq)[:, :, 0:-(self.kernel_size-1)*self.dilation]
        assert conv1d_out.shape == (B, self.out_channels, T)
        return F.leaky_relu(conv1d_out)

class GatedCausalConv1d(nn.Module):
    """Dilated causal conv with a GLU: the conv computes 2*channels, split
    into a value half and a gate half, output = v * sigmoid(g). The value
    path is linear (no tanh) -- Dauphin et al.'s GCNN result."""

    def __init__(self, kernel_size: int, channels: int, dilation: int=1):
        super(GatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1d = nn.Conv1d(channels, 2 * channels, kernel_size,
                                stride=1,
                                padding=(kernel_size - 1) * dilation,
                                dilation=dilation)

    def forward(self, seq: Tens):
        B, C, T = seq.shape
        out = self.conv1d(seq)[:, :, 0:-(self.kernel_size - 1) * self.dilation]
        v, g = out.chunk(2, dim=1)
        assert v.shape == (B, C, T)
        return v * torch.sigmoid(g)


class ChannelNorm(nn.Module):
    """LayerNorm over the channel dim of a (B, C, T) tensor."""

    def __init__(self, channels: int):
        super(ChannelNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tens):
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class GatedBlock(nn.Module):
    """Pre-norm residual block with a skip tap: the residual stream stays
    un-normalized, and each block emits a 1x1-projected skip contribution
    for the head to aggregate."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super(GatedBlock, self).__init__()
        self.norm = ChannelNorm(channels)
        self.conv = GatedCausalConv1d(kernel_size, channels, dilation)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tens):
        h = self.conv(self.norm(x))
        return x + h, self.skip(h)


class GatedDilationNet(nn.Module):
    """WaveNet-style stack: dilations kernel_size**h, and the output is the
    normalized sum of every block's skip tap rather than the top of the
    stack -- a path-length-1 gradient to each depth."""

    def __init__(self, channels: int, height: int, kernel_size: int):
        super(GatedDilationNet, self).__init__()
        self.blocks = nn.ModuleList(
            [GatedBlock(channels, kernel_size, dilation=kernel_size ** h)
             for h in range(height)])
        self.out_norm = ChannelNorm(channels)
        self.height = height

    def forward(self, x: Tens):
        total = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            total = total + skip
        return self.out_norm(total)


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super(Residual, self).__init__()
        self.submodule = submodule
        self.layer_norm = nn.LayerNorm(cast(int, submodule.out_channels))

    def forward(self, x: Tens):
        sum = x + self.submodule(x)
        # (B,C,T) -> (B,T,C)
        sum = sum.permute(0, 2, 1)
        n = self.layer_norm(sum)
        # (B,T,C) -> (B,C,T)
        return n.permute(0, 2, 1)


class DilationNet(nn.Module):
    def __init__(self, channels: int, height: int, kernel_size: int):
        super(DilationNet, self).__init__()
        self.layers = [ Residual(conv1d_factory(kernel_size, channels, channels, dilation=kernel_size ** h)) for h in range(height) ]
        self.net = nn.Sequential(*self.layers)
        self.height = height
    
    def forward(self, x: Tens):
        return self.net(x)
    
    def convs(self):
        for c in self.layers:
            yield c

class _TransformerBlock(nn.Module):
    """Pre-norm GPT-2-style block: causal self-attention + 4x GELU MLP."""

    def __init__(self, d: int, n_heads: int):
        super(_TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x: Tens):
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (B, T, self.n_heads, C // self.n_heads)
        q, k, v = (t.view(*shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.norm2(x))


class TransformerNet(nn.Module):
    """GPT-2-style transformer behind the same (B, C, T) interface as the
    conv stacks -- the attention baseline (arch 't1'). `height` is the
    layer count here."""

    def __init__(self, channels: int, n_layers: int):
        super(TransformerNet, self).__init__()
        n_heads = max(1, channels // 64)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(channels, n_heads) for _ in range(n_layers)])
        self.out_norm = ChannelNorm(channels)
        self.height = n_layers

    def forward(self, x: Tens):
        y = x.permute(0, 2, 1)
        for block in self.blocks:
            y = block(y)
        return self.out_norm(y.permute(0, 2, 1))


class SummNet(pl.LightningModule):
    def __init__(self,
                 vocab_size: int=29000,
                 dim: int=384,
                 fc_dim: int=1024,
                 height: Optional[int]=None,
                 max_length: int=2**20,
                 kernel_size: int=4,
                 pad_token_id: int=0,
                 lr: float=3e-4,
                 warmup_steps: int=1000,
                 lr_decay_steps: int=100_000,
                 arch: str='v1',
                 window: int=8,
                 cycles: int=3):
        # Layer h has dilation kernel_size**h, so a stack of H layers sees
        # kernel_size**H tokens back. Deriving H from max_length (the default)
        # gives the smallest stack that covers the whole context; anything
        # taller convolves nothing but left-padding. (For the transformer
        # baseline, height is just the layer count and kernel_size is unused.)
        if arch == 't1':
            if height is None:
                height = 12
        elif arch in ('v4', 'v4m'):
            # Depth comes from the dilation ladder: one cycle of levels
            # w^0..w^(L-1) covers window**L context, repeated `cycles` times.
            levels = max(1, math.ceil(math.log(max_length, window)))
            height = levels * cycles
        elif height is None:
            height = max(1, math.ceil(math.log(max_length, kernel_size)))
        elif kernel_size ** (height - 1) >= max_length:
            warnings.warn(
                f"height={height} overshoots: with kernel_size={kernel_size} "
                f"the top layer's dilation ({kernel_size ** (height - 1)}) is >= "
                f"max_length ({max_length}), so its long-range taps only ever "
                f"see padding. height={math.ceil(math.log(max_length, kernel_size))} "
                "already covers the context.")
        super(SummNet, self).__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_length = max_length
        self.total_train_tokens = 0
        self.fwd_flops_per_token = flops.fwd_flops_per_token(self.hparams)
        # 'v1' is the pre-2026-08-27 architecture; checkpoints saved before
        # the arch hparam existed default to it and keep loading.
        assert arch in ('v1', 'v2', 'v3', 'v4', 'v4m', 't1'), f"unknown arch {arch!r}"
        # Embed(B, T) -> (B, C, T)
        self.token_embedding_table = nn.Embedding(vocab_size, dim)
        if arch in ('v3', 'v4', 'v4m'):
            # No positional embedding: convolutions are translation-equivariant,
            # so the kernel structure already encodes relative position, and
            # packed windows make absolute position-in-window arbitrary anyway.
            # (v4's slot biases carry offset identity the same way kernels do.)
            # Also frees inference from the max_length cap.
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Parameter(0.1 * torch.randn( (dim, max_length)).to(self.device))
        if arch == 't1':
            self.filter_bank = TransformerNet(dim, height)
        elif arch in ('v4', 'v4m'):
            self.filter_bank = TreeAttentionNet(dim, window,
                                                levels=height // cycles,
                                                cycles=cycles,
                                                mlp=(arch == 'v4m'))
        elif arch in ('v2', 'v3'):
            self.filter_bank = GatedDilationNet(dim, height, kernel_size)
        else:
            self.filter_bank = DilationNet(dim, height, kernel_size)
        # No activation after the final Linear: cross-entropy wants unbounded
        # logits, and squashing the negative half keeps the model from ever
        # confidently ruling a token out.
        self.head = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_dim, vocab_size),
        )
        if arch in ('v2', 'v3', 'v4', 'v4m', 't1'):
            # Weight tying pays for the wider gated convs; needs fc_dim == dim
            # so the tied matrix has the embedding's (vocab, dim) shape.
            assert fc_dim == dim, "weight tying requires fc_dim == dim"
            self.head[-1].weight = self.token_embedding_table.weight
            # The tied matrix is also the output projection, and nn.Embedding's
            # default N(0,1) init is ~30x hotter than a Linear's -- the model
            # then spends thousands of warmup steps unlearning oversized
            # logits (measured: +0.77 nats val at step 2.5k). GPT-style scale.
            nn.init.normal_(self.token_embedding_table.weight, std=0.02)
        self.gc_time = dt.datetime.now()

    def forward(self, xi: Tens, _=None):
        x = self.token_embedding_table(xi).transpose(1, 2)
        B, C, T = x.shape

        if self.pos_embedding is not None:
            # Absolute PE hard-limits sequence length; without it (v3) any
            # length the receptive field can use is fine.
            assert T <= self.max_length
            x = x + self.pos_embedding[:, :T]
        filt = self.filter_bank(x)
        assert filt.shape == x.shape
        ## Segregate time channels by bouncing B,T into the 0'th dimension
        filt_trans = filt.transpose(1, 2)
        assert filt_trans.shape == (B, T, C)
        filt_trans = filt_trans.reshape(B * T, C)
        assert filt_trans.shape == (B * T, C)
        y_hat = self.head(filt_trans)
        assert y_hat.shape == (B * T, self.vocab_size)
        return y_hat
    
    def _shared_eval(self, batch: Tens, batch_idx: int, prefix: str):
        batch = batch.to(self.device)
        self._defrag()
        
        B, T = batch.shape
        assert T <= self.max_length

        x = batch[:, :-1]
        y = batch[:, 1:]
        assert y.shape == (B, T - 1)
        assert x.shape == (B, T - 1)
        y_hat = self(x)
        assert y_hat.shape == (B * (T - 1), self.vocab_size)
        # ignore_index: every position after a document's end is a pad target,
        # and without this the model spends gradient learning to predict [PAD].
        # The mean is then over real tokens only, so losses stay comparable
        # across batches with different amounts of padding.
        loss = F.cross_entropy(y_hat, y.reshape(-1),
                               ignore_index=self.hparams.pad_token_id)
        self.log(prefix + '_loss', loss, prog_bar=True)
        self.log('length', 1.0 * T)
        # A running total, not something to average over the epoch -- hence
        # log_metrics rather than self.log. There isn't always a logger to talk
        # to (bare training_step calls in tests, `--logger none`). Throttled:
        # a row per step is 55k inserts/run of pure metrics-DB churn.
        if self.logger is not None and batch_idx % 100 == 0:
            self.logger.log_metrics({
                'train_tokens': self.total_train_tokens,
                # 3x forward: the usual fwd+bwd training approximation.
                # Enables loss-vs-compute plots across architectures.
                'train_gflops': 3.0 * self.fwd_flops_per_token
                                * self.total_train_tokens / 1e9,
            })
        return loss

    def _defrag(self):
        if dt.datetime.now() - self.gc_time > dt.timedelta(seconds=90):
            util.defrag_cuda_memory()
            torch.cuda.empty_cache()
            self.gc_time = dt.datetime.now()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=1e-2)

        # Explicit decay horizon rather than trainer.estimated_stepping_batches:
        # streaming datasets have no length, so Lightning can't estimate one.
        warmup = max(1, self.hparams.warmup_steps)
        decay = max(warmup + 1, self.hparams.lr_decay_steps)

        def lr_factor(step: int):
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / (decay - warmup))
            # Cosine from the peak down to a 10% floor, then hold.
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def training_step(self, batch: Dict[str, Tens], batch_idx: int):
        # int(), not the raw tensor: keeps the counter off the GPU and keeps it
        # a plain scalar the loggers can serialize.
        self.total_train_tokens += int(torch.sum(batch['num_tokens']))
        return self._shared_eval(batch['input_ids'], batch_idx, 'train')
    
    def validation_step(self, batch: Dict[str, Tens], batch_idx: int):
        return self._shared_eval(batch['input_ids'], batch_idx, 'val')

    def test_step(self, batch: Dict[str, Tens], batch_idx: int):
        return self._shared_eval(batch['input_ids'], batch_idx, 'test')


def generate(model: pl.LightningModule, idx: Tens, max_new_tokens: int=100):
    idx = idx.to(model.device)
    assert(idx.dim() == 2)
    # Accumulate predicted tokens here. XXX: could just chop off tail of idx instead
    preds = idx.clone().to(model.device).squeeze(dim=0)

    for _ in range(max_new_tokens):
        # forward() returns (B*T, V), flat in time -- so with B == 1 the last
        # row is the prediction for the next token. (Indexing this as if it were
        # (B, T, V) raises IndexError.)
        logits = model(idx)[-1, :]  # Only care about the last prediction
        probs = F.softmax(logits, dim=-1)
        pred_y = torch.multinomial(probs, 1)
        preds = torch.cat( (preds, pred_y), 0)
        idx = preds.unsqueeze(0)
        if pred_y[0] == 102: # XXX: hardcoded EOS token
            break
    return preds

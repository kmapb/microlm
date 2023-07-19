import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import util
import datetime as dt

__CUDA__ = torch.cuda.is_available()

class CausalConv1d(nn.Module):
    """
    A causal 1D convolution.
    """
    def __init__(self, kernel_size, in_channels, out_channels, dilation=1):
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

    def forward(self, seq):
        """
        Expects a 3D tensor of shape (batch_size, channels, seq_len).
        """
        B,C,T = seq.shape
        assert(C == self.in_channels)
        #print("QQQ {}, dil {}".format(seq.shape, self.dilation))
        conv1d_out = self.conv1d(seq)[:, :, 0:-(self.kernel_size-1)*self.dilation]
        assert conv1d_out.shape == (B, self.out_channels, T)
        return F.leaky_relu(conv1d_out)

class Residual(nn.Module):
    def __init__(self, submodule):
        super(Residual, self).__init__()
        self.submodule = submodule
        self.layer_norm = nn.LayerNorm(submodule.out_channels)

    def forward(self, x):
        sum = x + self.submodule(x)
        # (B,C,T) -> (B,T,C)
        sum = sum.permute(0, 2, 1)
        n = self.layer_norm(sum)
        # (B,T,C) -> (B,C,T
        return n.permute(0, 2, 1)


class DilationNet(nn.Module):
    def __init__(self, channels, height):
        super(DilationNet, self).__init__()
        self.layers = [ Residual(CausalConv1d(2, channels, channels, dilation=2 ** h)) for h in range(height) ]
        self.net = nn.Sequential(*self.layers)
        self.height = height
    
    def forward(self, x):
        return self.net(x)
    
    def convs(self):
        for c in self.layers:
            yield c

class SummNet(pl.LightningModule):
    def __init__(self, vocab_size=29000, dim=384, fc_dim=1024, height=16, max_length=2**17):
        super(SummNet, self).__init__()
        self.dim = dim
        self.max_length = max_length
        self.save_hyperparameters()
        # Embed(B, T) -> (B, C, T)
        self.token_embedding_table = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn( (dim, max_length) )).to(self.device)
        self.filter_bank = DilationNet(dim, height)
        self.head = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_dim, vocab_size),
            nn.LeakyReLU(),
        )
        self.gc_time = dt.datetime.now()

    def forward(self, xi, _=None):
        x = self.token_embedding_table(xi).transpose(1, 2)
        B, C, T = x.shape
        
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
        assert y_hat.shape == (B * T, self.hparams.vocab_size)
        return y_hat
    
    def _shared_eval(self, batch, batch_idx, prefix):
        self._defrag()
        
        B, T = batch.shape
        assert T <= self.max_length

        x = batch[:, :-1]
        y = batch[:, 1:]
        assert y.shape == (B, T - 1)
        assert x.shape == (B, T - 1)
        y_hat = self(x)
        assert y_hat.shape == (B * (T - 1), self.hparams.vocab_size)
        loss = F.cross_entropy(y_hat, y.reshape(-1))
        self.log(prefix + '_loss', loss, prog_bar=True)
        self.log('length', 1.0 * T)
        self.log(prefix + 'cuda_malloc_mb', torch.cuda.memory_allocated(0)/1024.0/1024)
        self.log(prefix + 'cuda_reserved_mb', torch.cuda.memory_reserved(0)/1024.0/1024)
        self.log(prefix + 'cuda_max_reserved_mb', torch.cuda.max_memory_reserved(0)/1024.0/1024)
        return loss

    def _defrag(self):
            if dt.datetime.now() - self.gc_time > dt.timedelta(seconds=90):
                util.defrag_cuda_memory()
                torch.cuda.empty_cache()
                self.gc_time = dt.datetime.now()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-2)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'test')


def generate(model, idx=None, max_new_tokens=100):    
    if idx == None:
        idx = torch.zeros( (1, 1), dtype=torch.long, device=model.device )
    idx = idx.to(model.device)
    assert(idx.dim() == 2)
    # Accumulate predicted tokens here. XXX: could just chop off tail of idx instead
    preds = idx.clone().to(model.device).squeeze(dim=0)

    for _ in range(max_new_tokens):
        logits = model(idx)[0, -1, :] # Only care about the last prediction
        probs = F.softmax(logits, dim=-1)
        pred_y = torch.multinomial(probs, 1)
        preds = torch.cat( (preds, pred_y), 0)
        idx = preds.unsqueeze(0)
        if pred_y[0] == 102: # XXX: hardcoded EOS token
            break
    return preds

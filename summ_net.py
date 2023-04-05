import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from util import dev

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
        # print("QQQ {}, dil {}".format(seq.shape, self.dilation))
        conv1d_out = self.conv1d(seq)[:, :, 0:-(self.kernel_size-1)*self.dilation]
        assert conv1d_out.shape == (B, self.out_channels, T)
        return F.leaky_relu(conv1d_out)

class DilationNet(nn.Module):
    def __init__(self, channels, height):
        super(DilationNet, self).__init__()
        self.layers = [ CausalConv1d(2, channels, channels, dilation=2 ** h) for h in range(height) ]
        self.net = nn.Sequential(*self.layers)
        self.height = height
    
    def forward(self, x):
        return self.net(x)
    
    def convs(self):
        for c in self.layers:
            yield c
    
class SummNet(pl.LightningModule):
    def __init__(self, vocab_size=29000, dim=384, fc_dim=1024, height=16):
        super(SummNet, self).__init__()
        self.dim = dim
        self.save_hyperparameters()
        # Embed(B, T) -> (B, C, T)
        self.token_embedding_table = nn.Embedding(vocab_size, dim,
                                                  scale_grad_by_freq=True,
                                                  max_norm=0.2)
        self.filter_bank = DilationNet(dim, height)
        self.head = nn.Sequential(
            nn.Linear(dim, fc_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_dim, vocab_size),
            nn.LeakyReLU(),
        )

    def forward(self, xi, _=None):
        x = self.token_embedding_table(xi).transpose(1, 2)
        B, C, T = x.shape
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
        B, T = batch.shape
        if False:
            if T == 2:
                # Nothing to learn here! Return zero loss.
                z = torch.ones(1, requires_grad=True)
                return F.cross_entropy(z, z)
        x = batch[:, :-1]
        y = batch[:, 1:]
        assert y.shape == (B, T - 1)
        assert x.shape == (B, T - 1)
        y_hat = self(x)
        assert y_hat.shape == (B * (T - 1), self.hparams.vocab_size)
        loss = F.cross_entropy(y_hat, y.reshape(-1))
        self.log(prefix + '_loss', loss)
        self.log('length', 1.0 * T)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_eval(batch['input_ids'], batch_idx, 'test')


def generate(model, idx=None, max_new_tokens=100):    
    if idx == None:
        idx = torch.zeros( (1, 1), dtype=torch.long ).to(dev())
    idx = idx.to(dev())
    assert(idx.dim() == 2)
    # Accumulate predicted tokens here. XXX: could just chop off tail of idx instead
    preds = idx.clone().to(dev()).squeeze(dim=0)

    for _ in range(max_new_tokens):
        logits = model(idx)[0, -1, :] # Only care about the last prediction
        probs = F.softmax(logits, dim=-1)
        pred_y = torch.multinomial(probs, 1)
        preds = torch.cat( (preds, pred_y), 0)
        idx = preds.unsqueeze(0)
        if pred_y[0] == 102: # XXX: hardcoded EOS token
            break
    return preds

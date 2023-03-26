import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from util import dev

class FilterBank(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 filter_depth):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.filter_depth = filter_depth
        
        super(FilterBank, self).__init__()
        self.f = nn.Sequential(
          nn.Conv1d(channels_in, channels_out, filter_depth, padding='same', padding_mode='replicate'),
          nn.LocalResponseNorm(3),
          nn.MaxPool1d(2),
          nn.BatchNorm1d(channels_out),
          nn.ReLU())

    def input_size(self, input_length):
        return (self.channels_in, input_length)

    def output_size(self, input_length):
        return (self.channels_out,
               int((1 + input_length - self.filter_depth) / 2))
    
    def forward(self, x):
        return self.f(x)

class ReConvText(pl.LightningModule):
    def __init__(self, vocab_size=29000, filter_width=5, dim=384, fc_dim=1024):
        super(ReConvText, self).__init__()
        self.save_hyperparameters()

        self.dim = dim
        self.fc_dim = fc_dim
        self.token_embedding_table = nn.Embedding(vocab_size, dim)
        self.filter_bank = FilterBank(dim, dim, filter_width)
        self.head = nn.Sequential(
            nn.LayerNorm(fc_dim, eps=1e-6),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.LayerNorm(fc_dim, eps=1e-6),
            nn.Linear(fc_dim, vocab_size),
            nn.ReLU(),
        )

    def forward(self, xi, _=None):    
        x = self.token_embedding_table(xi).transpose(1, 2)
        while True:
            B, C, T = x.shape
            assert C == self.dim
            if T * C < self.fc_dim: # Very short snips: straight to FC!
                break
            x = self.filter_bank(x)
        x = F.pad(x.flatten(start_dim = 1), (0, self.fc_dim - T * C))
        assert x.shape == (B, self.fc_dim)
        return self.head(x)
    
    def _shared_eval(self, batch, batch_idx, prefix):
        B, T = batch.shape
        x = batch[:, :-1]
        y = batch[:, -1]
        assert y.shape == (B,)
        assert x.shape == (B, T - 1)
        y_hat = self(x)
        assert y_hat.shape == (B, self.hparams.vocab_size)
        loss = F.cross_entropy(y_hat, y)
        self.log(prefix + '_loss', loss)
        self.log('length', 1.0 * T)
        return loss
    
    def training_step(self, batch, batch_idx):
        B, T = None, None
        if len(batch.shape) == 2:
            B, T = batch.shape
        else:
            B, T = (1, batch.shape[0])
        if T == 2:
            # Nothing to learn here: START/END.
            z = torch.ones(1, requires_grad=True)
            return F.cross_entropy(z, z)
        return self._shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

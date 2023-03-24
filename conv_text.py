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
    
    def training_step_one(self, x, y):
        return F.cross_entropy(self(x), y)
    
    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        y = batch
        B, T = x.shape
        assert y.shape == (B, T)
        loss = 0.0
        max_samples=100

        if True:
            n = 0
            if T > max_samples:
                for i in range(0, max_samples):
                    i = torch.randint(low=1, high=T-1, size=()).item()
                    n  += 1
                    loss += self.training_step_one(x[:, :i], y[:, i])
            else:
                for i in range(1, T):
                    n += 1
                    loss += self.training_step_one(x[:, :i-1], y[:, i-1])
            self.log(prefix + '_loss', loss/n)
            self.log('length', 1.0 * T)
            self.log('2ndchar', 1.0 * y[0, 1])
        else:
            import pdb; pdb.set_trace()
            loss = self.training_step_one(x[:, :T-1], y[:, T-1])
            print(loss)
            self.log(prefix + '_loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
        
class FilterApparatus(nn.Module):
    def __init__(self, embedding_width):
        super(FilterApparatus, self).__init__()

        self.fb1 = FilterBank(embedding_width, 1024, 3)
        self.fb2 = FilterBank(1024, 1024, 4)
        self.fb3 = FilterBank(1024, 1024, 5)
        self.fb4 = FilterBank(1024, 1024, 5)
        self.fb5 = FilterBank(1024, 64, 5)
        self.apparatus = nn.Sequential(
            self.fb1, self.fb2, self.fb3, self.fb4, self.fb5)

    def input_size(self, input_width, input_length):
        return self.fb1.input_size(input_width, input_length)
                                   
    def output_size(self, input_length):
        fb1_sz = self.fb1.output_size(input_length)
        fb2_sz = self.fb2.output_size(fb1_sz[1])
        fb3_sz = self.fb3.output_size(fb2_sz[1])
        fb4_sz = self.fb4.output_size(fb3_sz[1])
        return self.fb5.output_size(fb4_sz[1])
    
    def forward(self, x):
        return self.apparatus(x)

class ConvText(pl.LightningModule):
    def __init__(self,
                 vocab_size = 29000,
                 embedding_width=100,
                 context_size=8192):
        super(ConvText, self).__init__()
        self.save_hyperparameters()

        self.context_length = context_size
        self.embedding_width = embedding_width

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_width).to(dev())
        self.pos_embedding_table = nn.Embedding(context_size, embedding_width).to(dev())
        self.pos_vector = torch.tensor(range(context_size), dtype=torch.int32).to(dev())

        self.filter_stack = FilterApparatus(embedding_width)
        filter_out_size = self.filter_stack.output_size(context_size)
        filter_out_nparams = filter_out_size[0] * filter_out_size[1]
        print("first FC has {} inputs ({})".format(filter_out_nparams, filter_out_size))
        self.model = nn.Sequential(
            self.filter_stack,
            nn.Flatten(),
            nn.Linear(filter_out_nparams, 8192),
            nn.ReLU(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Linear(8192, vocab_size)).to(dev())

    def _project(self, x):
        padded_x = torch.zeros((x.shape[0], self.context_length), dtype=torch.long, device=dev())

        padded_x[:, -x.shape[1]:] = x
        assert(padded_x.shape[1] == self.context_length)
        projected_toks = self.token_embedding_table(padded_x)
        projected_pos = self.pos_embedding_table(self.pos_vector)
        projected_pos = projected_pos.expand(x.shape[0], self.context_length, self.embedding_width)
        projected = projected_toks + projected_pos
        # Conv layer expects (B,C,T), so:
        projected = projected.transpose(1, 2)
        assert len(projected_toks.size()) == 3
        assert projected.size()[0] == x.size()[0] # B
        assert projected.size()[1] == self.embedding_width # C
        assert projected.size()[2] == self.context_length # T
        return projected

    def forward(self, x):
        return self.model(self._project(x))
    
    def training_step_one(self, x, y):
        y_hat = self(x)
        B, T = y_hat.shape
        y = y.view(B)
        return F.cross_entropy(y_hat, y)
    
    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch['input_ids']
        y = batch['labels']
        _, T = x.shape
        # Pick a random prefix length. Bias? Eh.
        loss = 0.0
        n = 0
        lengths = [1, 8, 64, 256, 4007, 8013]
        for i in lengths:
            if i >= T:
                break
            n += 1
            loss += self.training_step_one(x[:, :i], y[:, i])
        self.log(prefix + '_loss', loss / n)
        return loss
    
    def training_step(self, batch, batch_idx):
        if True:
            self.log("training_step", batch[0][1])
        return self._shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            return self._shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def generate(self, idx=None, max_new_tokens=100):    
        if idx == None:
            idx = torch.zeros( (1, 1), dtype=torch.long ).to(dev())
        idx = idx.to(dev())
        assert(idx.dim() == 2)
        # Accumulate predicted tokens here. XXX: could just chop off tail of idx instead
        preds = torch.zeros(idx.shape[0], 0).to(dev())

        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            probs = F.softmax(logits, dim=-1)
            pred_y = torch.multinomial(probs, 1)
            preds = torch.cat( (preds, pred_y), 1)
        return preds[0]


def generate(model, idx=None, max_new_tokens=100):    
    if idx == None:
        idx = torch.zeros( (1, 1), dtype=torch.long ).to(dev())
    idx = idx.to(dev())
    assert(idx.dim() == 2)
    # Accumulate predicted tokens here. XXX: could just chop off tail of idx instead
    preds = torch.zeros(idx.shape[0], 0).to(dev())

    for _ in range(max_new_tokens):
        logits = model(idx)
        probs = F.softmax(logits, dim=-1)
        pred_y = torch.multinomial(probs, 1)
        preds = torch.cat( (preds, pred_y), 1)
    return preds[0]


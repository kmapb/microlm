import torch
from util import dev
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

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
          nn.Conv1d(channels_in, channels_out, filter_depth),
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
        if True:
            return self.apparatus(x)
        for fb in [self.fb1, self.fb2, self.fb3, self.fb4, self.fb5]:
            y = fb(x)
            print("x.shape {}, y.shape {}".format(x.shape, y.shape))
            x = y
        return y


class ConvText(pl.LightningModule):
    def __init__(self,
                 vocab_size = 29000,
                 embedding_width=100,
                 context_size=8192):
        super(ConvText, self).__init__()
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
    
    def training_step(self, batch, batch_idx):
        import pdb; pdb.set_trace()
        x, y = batch
        y_hat = self(x)
        B, T = y_hat.shape
        y = y.view(B)
        print(y_hat.shape, y.shape)
        return F.cross_entropy(y_hat, y)

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

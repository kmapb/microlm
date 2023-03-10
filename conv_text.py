import torch
from util import dev
from torch import nn
from torch.nn import functional as F


class FilterBank(nn.Module):
    def __init__(self, channels_in, channels_out, width, filter_depth):
        super(FilterBank, self).__init__()
        self.f = nn.Sequential(
          nn.Conv2d(channels_in, channels_out, (width, filter_depth)),
          nn.MaxPool2d((1, 2)),
          nn.ReLU())

    def forward(self, x):
        return self.f(x)

class FilterApparatus(nn.Module):
    def __init__(self, embedding_width):
        super(FilterApparatus, self).__init__()
        self.fb1 = FilterBank(1, 1024, embedding_width, 15)
        self.fb2 = FilterBank(1024, 512, 1, 15)
        self.fb3 = FilterBank(512, 128, 1, 15)
        self.fb4 = FilterBank(128, 64, 1, 15)
        self.fb5 = FilterBank(64, 64, 1, 15)
        self.apparatus = nn.Sequential(
            self.fb1, self.fb2, self.fb3, self.fb4, self.fb5)
    
    def forward(self, x):
        return self.apparatus(x)

class ConvText(nn.Module):
    def __init__(self, vocab_size, embedding_width=384,
            n_filters = 32,
            filter_depth = 5):
        super(TokenRNNLM, self).__init__()

        self.hidden_size = hidden_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_width)
        self.i2h = nn.Sequential(
          nn.LayerNorm((embedding_width + hidden_size)),
          nn.Linear(embedding_width + hidden_size, hidden_size),
          nn.LeakyReLU())
        self.i2o = nn.Sequential(
          nn.Linear(embedding_width + hidden_size, vocab_size),
          nn.LeakyReLU())

    def forward(self, idx, hidden=None, targets=None):
        assert(idx.dim() == 2)
        if hidden is None:
            hidden = torch.zeros(idx.shape + (self.hidden_size,)).to(dev())
        # input is sparse character indices, shaped (B,T,C)
        combined = torch.cat( (self.token_embedding_table(idx), hidden), 2)
        hidden = self.i2h(combined)
        logits = self.i2o(combined)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, hidden, loss
      
    def generate(self, idx=None, max_new_tokens=100):
        if idx == None:
            idx = torch.zeros( (1, 1), dtype=torch.long ).to(dev())
        idx = idx.to(dev())
        # idx is (B, T) array of indices in the current context
        assert(idx.dim() == 2)
        assert(idx.shape[1] >= 1)
        # Queue up the right hidden state by playing through idx
        h = None
        logits = None
        loss = None

        for i in range(idx.shape[1]):
            logits, h, loss = self(idx.select(1, i)[None, :], h)

        preds = torch.zeros(idx.shape[0], 0).to(dev())
        for _ in range(max_new_tokens):
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            preds = torch.cat((preds, idx), dim=1) # (B, T+1)
            # get the next predictions
            logits, h, loss = self(idx, h)
        return preds[0].tolist()

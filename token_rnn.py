import torch
from util import dev
from torch import nn

class TokenRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_width=384, hidden_size=1024):
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

        print(hidden)
        print(idx)
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
      
    def generate(self, idx, max_new_tokens):
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
        return preds

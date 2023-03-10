import torch
from util import dev
from torch import nn
from torch.nn import functional as F


class FilterBank(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 filter_width,
                 filter_depth):
        self.channels_out = channels_out
        self.filter_width = filter_width
        self.filter_depth = filter_depth
        
        super(FilterBank, self).__init__()
        self.f = nn.Sequential(
          nn.Conv2d(channels_in, channels_out, (filter_width, filter_depth)),
          nn.MaxPool2d((1, 2)),
          nn.Dropout(),
          nn.ReLU())

    def input_size(self, input_width, input_length):
        return (1, input_width, input_length)

    def output_size(self, input_width, input_length):
        return (self.channels_out,
                1 + (input_width - self.filter_width),
                int((input_length - self.filter_depth + 1) / 2))
    
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

    def input_size(self, input_width, input_length):
        return self.fb1.input_size(input_width, input_length)
                                   
    def output_size(self, input_width, input_length):
        fb1_sz = self.fb1.output_size(input_width, input_length)
        fb2_sz = self.fb2.output_size(fb1_sz[1], fb1_sz[2])
        fb3_sz = self.fb3.output_size(fb2_sz[1], fb2_sz[2])
        fb4_sz = self.fb4.output_size(fb3_sz[1], fb3_sz[2])
        return self.fb5.output_size(fb4_sz[1], fb4_sz[2])
    
    def forward(self, x):
        return self.apparatus(x)

class ConvText(nn.Module):
    def __init__(self,
                 vocab_size = 29000,
                 embedding_width=384,
                 context_size=1024):
        super(ConvText, self).__init__()
        self.context_length = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_width).to(dev())
        self.filter_stack = FilterApparatus(embedding_width)
        filter_out_size = self.filter_stack.output_size(embedding_width, context_size)
        filter_out_nparams = filter_out_size[0] * filter_out_size[1] * filter_out_size[2]
        self.model = nn.Sequential(
            self.filter_stack,
            nn.Flatten(),
            nn.Linear(filter_out_nparams, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, 8192),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8192, vocab_size)).to(dev())

    def forward(self, idx, targets=None):
        assert(idx.dim() == 2) # B,Tokens
        padded_idx = torch.zeros((idx.shape[0], self.context_length), dtype=torch.long ).to(dev())
        padded_idx[:, -idx.shape[1]:] = idx
        assert(padded_idx.shape[1] == self.context_length)
        # input is sparse character indices, shaped (B,T,C)
        projected = self.token_embedding_table(padded_idx)
        logits = self.model(projected)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
      
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
        return preds

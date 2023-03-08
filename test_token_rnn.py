import torch
import text_data
import token_rnn
from util import dev

if __name__ == "__main__":
    rnn = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
    print(rnn.generate(torch.zeros( (2, 2), dtype=torch.int32).to(dev()), 12))

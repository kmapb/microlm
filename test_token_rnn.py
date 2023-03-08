import torch
import text_data
import token_rnn

from util import dev

if __name__ == "__main__":
    rnn = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
    t = rnn.generate(idx=torch.zeros( (1, 1), dtype=torch.long ).cuda(), max_new_tokens=15)
    print("Output: {}".format(text_data.decode(t[0].cpu().numpy())))

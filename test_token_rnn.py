import torch
import text_data
import token_rnn

from util import dev

if __name__ == "__main__":
    rnn = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
    t = rnn.generate()
    print("Output: {}".format(text_data.decode(t)))
    rnn2 = torch.load('model.pt')
    print("Output: {}".format(text_data.decode(rnn2.generate())))
    print(text_data.complete_prefix(rnn, 'The internet ', max_new_tokens=15))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
import text_data as td
from util import dev

class ConvTextModel(nn.Module):
    def __init__(self, num_tokens=29000, context_width=8192, embed_dim=256, num_filters=64, kernel_size=5):
        super(ConvTextModel, self).__init__()
        self.num_tokens = num_tokens # Number of tokens in the vocabulary
        self.embed_dim = embed_dim
        self.context_width = context_width

        # Learned positional encoding
        self.position_enc = nn.Embedding(self.context_width, self.embed_dim)
        # Constant pos indexing vector
        self.pos_vector = torch.tensor(range(self.context_width), dtype=torch.int32) \
            .to(dev())

        # Token embedding
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        # 1D Convolution layer
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size // 2)

        # Fully connected layer for the output
        self.fc = nn.Linear(num_filters, num_tokens)

    def _prep_inputs(self, x):
        if x.dim() == 1:
            x = x.reshape(1, -1) # Add batch dimension
        pad_length = self.context_width - x.shape[1]
        if pad_length > 0:
            x = torch.cat((x, torch.full((1, pad_length), td.pad_token_id(), dtype=torch.int32, device=x.device)), dim=1)
        return x.to(dev())
    
    def forward(self, x, targets=None):
        # Adding positional encoding
        x = self._prep_inputs(x)
        # print("prepped x shape: {}".format(x.shape))
        positions = self.pos_vector.repeat(x.size(0), 1)
        # print("x max/min: {} {}".format(x.max(), x.min()))
        assert x.max() >= 0 and x.min() >= 0
        assert x.min() < self.num_tokens and x.max() < self.num_tokens
        
        # print("pos max/min: {} {}".format(positions.max(), positions.min()))
        assert positions.max() >= 0 and positions.min() >= 0
        assert positions.max() < self.context_width and positions.min() < self.context_width
        pos_enc = self.position_enc(positions).expand(x.size(0), -1, -1)
        
        emb = self.token_embedding(x)
        # print("x/emb/pos_enc shape: {}/{}/{}".format(x.shape, emb.shape, pos_enc.shape))
        x = emb + pos_enc

        # Transpose the tensor to match the input shape of Conv1d
        x = x.transpose(1, 2)

        # Apply the convolution
        x = self.conv(x)

        # Apply the activation function (e.g., ReLU)
        x = torch.relu(x)

        # Transpose the tensor back to its original shape. Not needed for a FC layer next, but if
        # we were to do more Conv it would be important.
        x = x.transpose(1, 2)

        # Put this big layer into [B, C] format
        x = x.reshape((x.shape[0], -1))

        # Apply the fully connected layer
        x = self.fc(x)

        # Apply softmax to get probabilities
        x = torch.softmax(x, dim=-1)

        if targets is None:
            return x
        print("x shape: {}; targets shape: {}".format(x.shape, targets.shape))
        return x, F.cross_entropy(x, targets)

    def generate(self, prefix='', max_new_tokens=10, temperature=1.0):
        self.eval()
        with torch.no_grad():
            input_tokens = td.encode(prefix, add_special_tokens=False).reshape(1, -1).to(dev())
            for _ in range(max_new_tokens):
                x = self._prep_inputs(input_tokens)
                output = self(input_tokens)
                next_token_logits = output[:, -1, :] / temperature
                
                # Sample from the softmax distribution
                probabilities = torch.softmax(next_token_logits, dim=-1)
                # print(probabilities.shape)
                next_token = torch.multinomial(probabilities[0], 1).squeeze()
                # print("token: {}".format(next_token))
                # Add the new token to the input tokens
                input_tokens = torch.cat((input_tokens, next_token.unsqueeze(0).unsqueeze(1)), dim=1)

            # Decode the generated tokens
            return input_tokens.squeeze().tolist()


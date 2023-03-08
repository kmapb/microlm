import torch

def dev():
    if torch.cuda.is_available():
        return torch.device('cuda', 0)
    return torch.device('cpu')


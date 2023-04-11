import torch
import os

def dev():
    if os.getenv('TORCH_CPU_ONLY'):
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda', 0)
    return torch.device('cpu')


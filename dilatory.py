import torch
import torch.nn as nn
import util

def dilated_indices(T = 4, filter_width = 2, dilation_rate = 2):
    padded_T = T + 1
    out_indices = torch.zeros(filter_width * padded_T, dtype=torch.int64)
    for span in range(0, padded_T):
        i = span * filter_width
        for j in range(0, filter_width):
            out_indices[i + j] = span - (filter_width - j - 1) * dilation_rate
    out_indices[out_indices >= padded_T] = T
    out_indices[out_indices < 0] = T
    return out_indices.flatten()

class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DilatedConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                #padding=kernel_size // 2,
                                stride=kernel_size)
    
    def forward(self, x):
        ## Zero-pad; n'th element contains zeros
        B, C, T = x.shape
        x = torch.cat([x, torch.zeros(B, C, 1, device=x.device)], dim=2)
        idxs = dilated_indices(T, self.kernel_size, self.dilation_rate)
        swizzle = x.index_select(2, idxs)
        return self.conv1d(swizzle)[:, :, :T]

if __name__ == "__main__":
    idxs = dilated_indices(4, 4, 1)
    assert (idxs == torch.tensor([4, 4, 4, 0,
                                 4, 4, 0, 1,
                                 4, 0, 1, 2,
                                 0, 1, 2, 3,
                                 1, 2, 3, 4])).all()
    print(dilated_indices(4, 4, 1))
    print(dilated_indices(15, 3, 5))
    B = 1
    C = 10
    T = 15
    c1 = DilatedConv1D(C, C, 2, 8)
    x = c1.forward(torch.randn(B, C, T))
    print(x.shape)
    assert x.shape == (B, C, T)
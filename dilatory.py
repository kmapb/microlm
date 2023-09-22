import torch
import torch.nn as nn

def dilated_indices(T = 4, filter_width = 2, dilation_rate = 2):
    padded_T = T + 1
    out_indices = torch.zeros(filter_width * padded_T, dtype=torch.int64)
    for span in range(0, padded_T):
        i = span * filter_width
        for j in range(0, filter_width):
            out_indices[i + j] = span + j * dilation_rate
    out_indices[out_indices >= padded_T] = T
    return out_indices.flatten()

class DilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, max_length=int(1e6)):
        super(DilatedConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=kernel_size)
        self.dilated_indices = dilated_indices(max_length, kernel_size, dilation_rate)
    
    def forward(self, x):
        ## Zero-pad; n'th element contains zeros
        B, C, T = x.shape
        x = torch.cat([x, torch.zeros(B, C, 1, device=x.device)], dim=2)
        swizzle = x.index_select(2, self.dilated_indices)
        return self.conv1d(swizzle)

if __name__ == "__main__":
    print(dilated_indices(15, 3, 5))
    B = 1
    C = 10
    T = 15
    c1 = DilatedConv1D(C, C, 2, 8, max_length=T)
    x = c1.forward(torch.randn(1, 10, 15))
    print(x.shape)
    assert x.shape == (B, C, T + 1)
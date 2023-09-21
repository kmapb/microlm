import torch

def dilated_indices(T = 4, filter_width = 2, dilation_rate = 2):
    padded_T = T + 1
    out_indices = torch.zeros(filter_width * padded_T, dtype=torch.int64)
    for span in range(0, padded_T):
        i = span * filter_width
        for j in range(0, filter_width):
            out_indices[i + j] = span + j * dilation_rate
    out_indices[out_indices >= padded_T] = T
    return out_indices.flatten()

if __name__ == "__main__":
    print(dilated_indices(15, 3, 5))
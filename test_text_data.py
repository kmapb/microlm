import torch
import text_data as td

if __name__ == "__main__":
    enc = torch.tensor( range(1, 6), dtype=torch.long)
    x, y = td.embatch(enc)
    print(x)
    print(y)

    x, y = td.embatch(enc, max_batch_size=2)
    print(x)
    print(y)

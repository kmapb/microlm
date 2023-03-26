import sys
import torch
import pytorch_lightning as pl
import conv_text as ct
import text_data as td
from util import dev

def chat(mdl):
    while True:
        prompt = input('! ')
        toks = ct.generate(mdl, torch.unsqueeze(td.encode(prompt), 0).to(dev()))
        print(td.decode(toks))

if __name__ == "__main__":
    mdl = 'model.ckpt'
    if len(sys.argv) > 1:
        mdl = sys.argv[1]
    print("loading {} ...".format(mdl),)
    mdl = ct.ReConvText.load_from_checkpoint(mdl).to(dev())
    print("done")
    chat(mdl)


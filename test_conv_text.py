import torch
import conv_text as ct
from text_data import decode, encode
from util import dev

if __name__ == "__main__":
    ct1 = ct.ConvText(29000, 384, 8192)
    print("made it!")
    tt = encode("The internet is a ").to(dev())
    tt = tt[None, :]
    print(ct1(tt))
    ct1.filter_stack.fb1.forward(tt)
    genzo = ct1.generate()
    print("a bad prediction: {}".format(decode(genzo)))


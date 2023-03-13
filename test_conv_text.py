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
    genzo = ct1.generate()[0]
    print("genzo: {}".format(genzo))
    print(decode(genzo))
    # print("a bad prediction: {}".format(decode(genzo)))


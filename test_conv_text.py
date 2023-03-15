import torch
import conv_text as ct
import text_data
from text_data import decode, encode
from util import dev

if __name__ == "__main__":
    ct1 = ct.ConvText(text_data.vocabulary_size(), 100, 1024).to(dev())
    print("made it!")
    tt = encode("The internet is a ").to(dev())
    tt = tt[None, :]
    print(ct1(tt))
    genzo = ct1.generate()
    print("genzo: {}".format(genzo))
    print(decode(genzo))
    # print("a bad prediction: {}".format(decode(genzo)))


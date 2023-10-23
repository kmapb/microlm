import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import summ_net as sn
import text_data as td
import transformers
import math

from util import dev
from transformers import GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

def my_encode(s):
    tk = td.tokenize(s, add_special_tokens=False)
    return torch.tensor(tk).to(dev())

def respond(mdl, prompt, beam_size=15, max_new_tokens=100, temperature=1.1):
    idx =  torch.unsqueeze(my_encode(prompt), 0).to(dev())
    candidates = mdl.generate(
            idx,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=temperature,
            max_new_tokens=max_new_tokens)
    return td.decode(candidates[0])

def chat(mdl, beam_size, temp):
    with torch.no_grad():
        while True:
            print(respond(mdl, input('> '), beam_size=beam_size, temperature=temp))


class Generable(transformers.GenerationMixin, sn.SummNet):
    def __init__(self, *args, **kwargs):
        transformers.GenerationMixin.__init__(self)
        sn.SummNet.__init__(self, *args, **kwargs)
        
        # Support for HF framework stuff
        self.generation_config = GenerationConfig()
        self.config = PretrainedConfig(name='pb-alm')
        self.main_input_name = 'input_ids'

    def can_generate(self):
        return True
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids}
    
    def forward(self, input_ids=None, return_dict=False, **kwargs):
        # We're (B*T, V) here, but we want (B, T, V). Since B == 1 unsqueeze does it
        logits = sn.SummNet.forward(self, input_ids).unsqueeze(0)
        if return_dict:
            return SequenceClassifierOutput(logits=logits)
        return logits



if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog='chat.py',
        description='Synthesize text completions interactively',
        epilog='Beaming our way to your laptop\'s screen, and maybe ...  just maybe ... your heart'
    )
    parser.add_argument('filename')
    parser.add_argument('--beam-width', type=int, default=12,
            help='Beam width. Larger values run slower.')
    parser.add_argument('--temp', type=float, default=1.0,
            help='Temperature for estimator.')
    parser.add_argument('--greedy', type=bool, default=False,
            help='Greedy decoding')
    os.putenv('TORCH_CPU_ONLY', '1')
    mdl = 'model.ckpt'
    args = parser.parse_args()
    with torch.no_grad():
        mdl = Generable.load_from_checkpoint(args.filename).to(dev())
 
        print("loading {} ...".format(args.filename),)
        print("done")
        chat(mdl, args.beam_width, args.temp)

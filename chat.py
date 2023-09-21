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

def score(mdl, idx, sequence):
    logprobsum = 0.0
    for tok in sequence[idx.shape[1]:]:
        logits = mdl(idx)
        probs = F.softmax(logits, dim=-1)
        logprob = torch.log(probs[0, -1, tok])
        logprobsum += logprob
        idx = torch.cat( (idx, torch.tensor([[tok]]).to(dev())), dim=1)
    return logprobsum

def beamsearch(mdl, idx, beam_size=22, max_new_tokens=100, temperature=1.0):
    candidates = [ (0.0, False, idx) ]

    k_expansion_factor_per_beam = beam_size
    def dump_cands():
        i = 0
        print("------------------")
        for score, terminated, idx in candidates:
            print("{} Score: {}  Text: {}".format(i, score, td.decode(idx[0])))
            i += 1
        print("------------------")

    for tok in range(max_new_tokens):
        new_candidates = [] # Keep old ones in the running
        # XXX: vectorie this!
        for score, terminated, idx in candidates:
            if terminated:
                new_candidates.append( (score, terminated, idx) )
                continue
            # logits ~ B,T,V
            logits = mdl(idx)[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            preds = torch.multinomial(probs, k_expansion_factor_per_beam)
            for i in preds:
                logprob = torch.log(probs[i])
                # Apply a length-normalization; this is a hack
                new_score = (score * tok + logprob) / ((tok + 1) ** 0.8)
                new_candidates.append( (new_score,
                                        i == td.sep_token_id(),
                                        torch.cat( (idx, torch.tensor([[i]]).to(dev())), dim=1)) )
        candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        assert len(candidates) <= beam_size
        dump_cands()
    return candidates

def beamsearch_respond(mdl, prompt, beam_size=15, max_new_tokens=100, temperature=1.0):
    idx =  torch.unsqueeze(my_encode(prompt), 0).to(dev())
    candidates = beamsearch(mdl, idx,
                            beam_size=beam_size,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature)
    score,terminated,seq = candidates[0]
    return td.decode(seq[0])

def greedy_respond(mdl, prompt):
    idx = torch.unsqueeze(my_encode(prompt), 0).to(dev())
    toks = sn.generate(mdl, idx).to(dev())
    s = score(mdl, idx, toks)
    print("Score: {}".format(s))
    return td.decode(toks)

    d = { 'input_ids': idx, 'attention_mask': torch.ones_like(idx)}
    toks = mdl.generate(idx, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

def chat(mdl, beam_size, temp):
    with torch.no_grad():
        while True:
            print(beamsearch_respond(mdl, input('> '), beam_size=beam_size, temperature=temp))


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
    os.putenv('TORCH_CPU_ONLY', '1')
    mdl = 'model.ckpt'
    args = parser.parse_args()
    mdl = Generable.load_from_checkpoint(args.filename).to(dev())
    print("loading {} ...".format(args.filename),)
    print("done")
    chat(mdl, args.beam_width, args.temp)

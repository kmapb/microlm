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
    candidates = [ (-1e8, False, idx) ]

    k_expansion_factor_per_beam = beam_size
    def dump_cands():
        i = 0
        print("------------------")
        for score, terminated, idx in candidates:
            print("{} Score: {}  Text: {}".format(i, score, td.decode(idx[0])))
            i += 1
        print("------------------")

    for tok in range(1, max_new_tokens + 1):
        new_candidates = [] # Keep old ones in the running
        # XXX: vectorie this!
        for log_prob, terminated, idx in candidates:
            if terminated:
                new_candidates.append( (log_prob, terminated, idx) )
                continue
            # logits ~ B,T,V
            logits = mdl(idx)[:, -1, :] / temperature
            probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_preds = torch.topk(probs, k_expansion_factor_per_beam)
            for new_log_prob, new_token in zip(topk_log_probs[0], topk_preds[0]):
                new_log_prob += log_prob # Accumulate raw logprobs
                # Apply a length-normalization; this is a hack
                new_score = new_log_prob / (tok  ** 0.7)
                new_candidates.append( (new_score,
                                        new_token == td.sep_token_id(),
                                        torch.cat( (idx, torch.tensor([[new_token]]).to(dev())), dim=1)) )
        candidates = sorted(candidates + new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
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

def greedy_respond(mdl, prompt, beam_size=0, temperature=0.0):
    idx = torch.unsqueeze(my_encode(prompt), 0).to(dev())
    toks = sn.generate(mdl, idx).to(dev())
    s = score(mdl, idx, toks)
    print("Score: {}".format(s))
    return td.decode(toks)

    d = { 'input_ids': idx, 'attention_mask': torch.ones_like(idx)}
    toks = mdl.generate(idx, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

def chat(mdl, beam_size, temp):
    resp = beamsearch_respond
    with torch.no_grad():
        while True:
            print(resp(mdl, input('> '), beam_size=beam_size, temperature=temp))


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
    with torch.no_grad():
        mdl = Generable.load_from_checkpoint(args.filename).to(dev())

        print("loading {} ...".format(args.filename),)
        print("done")
        chat(mdl, args.beam_width, args.temp)

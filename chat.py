import sys
import torch
import pytorch_lightning as pl
import summ_net as sn
import text_data as td
import transformers
from util import dev
from transformers import GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

def my_encode(s):
    tk = td.tokenize(s, add_special_tokens=False)
    return torch.tensor(tk)

def respond(mdl, prompt):
    idx = torch.unsqueeze(my_encode(prompt), 0).to(dev())
    toks = sn.generate(mdl, idx).to(dev())
    return td.decode(toks)

    d = { 'input_ids': idx, 'attention_mask': torch.ones_like(idx)}
    toks = mdl.generate(idx, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

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


def chat(mdl):
    while True:
        print(respond(mdl, input('! ')))

if __name__ == "__main__":
    mdl = 'model.ckpt'
    if len(sys.argv) > 1:
        mdl = sys.argv[1]
    print("loading {} ...".format(mdl),)
    mdl = Generable.load_from_checkpoint(mdl).to(dev())
    print("done")
    chat(mdl)

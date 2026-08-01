import torch
import torch.nn.functional as F
import summ_net as sn
import text_data as td

from util import dev

def my_encode(s):
    tk = td.tokenize(s, add_special_tokens=False)
    return torch.tensor(tk, dtype=torch.long).to(dev())

def sample_token(logits, top_k=50, top_p=0.95, temperature=1.0):
    """
    Top-k then top-p (nucleus) sampling over a 1-D logits tensor.
    """
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][-1]
        logits = logits.masked_fill(logits < kth, -float('inf'))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Keep tokens until cumulative probability exceeds top_p (always
        # keeping at least the most likely one).
        drop = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, -float('inf'))
        logits = torch.full_like(logits, -float('inf')) \
            .scatter(0, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)

def respond(mdl, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
    idx = torch.unsqueeze(my_encode(prompt), 0)
    eos = td.sep_token_id()
    for _ in range(max_new_tokens):
        # forward() returns (B*T, V); with B == 1 the last row is the
        # next-token prediction.
        logits = mdl(idx)[-1, :]
        pred = sample_token(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        idx = torch.cat((idx, pred.unsqueeze(0)), dim=1)
        if pred.item() == eos:
            break
    return td.decode(idx[0])

def chat(mdl, temp, max_new_tokens):
    with torch.no_grad():
        while True:
            try:
                prompt = input('> ')
            except EOFError:
                break
            print(respond(mdl, prompt, max_new_tokens=max_new_tokens, temperature=temp))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='chat.py',
        description='Synthesize text completions interactively',
        epilog='Beaming our way to your laptop\'s screen, and maybe ...  just maybe ... your heart'
    )
    parser.add_argument('filename')
    parser.add_argument('--temp', type=float, default=1.0,
            help='Temperature for estimator.')
    parser.add_argument('--max-new-tokens', type=int, default=100,
            help='Maximum number of tokens to generate per prompt.')
    args = parser.parse_args()
    with torch.no_grad():
        print("loading {} ...".format(args.filename))
        mdl = sn.SummNet.load_from_checkpoint(args.filename, map_location=dev()).to(dev())
        mdl.eval()
        print("done")
        chat(mdl, args.temp, args.max_new_tokens)

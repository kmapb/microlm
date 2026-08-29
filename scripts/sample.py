"""Sample completions from a SummNet checkpoint with modern decoding.

The naive top-k sampler stopped being a fair witness once the routed
architectures (v4 family) grew strong copy circuits: teacher-forced loss
rewards copying, but a free-running greedy-ish decoder feeds the copy
circuit its own output and loops. Countermeasures here:

  * min-p truncation (keep tokens with p >= min_p * max_p): adapts the
    cutoff to the model's confidence, degrades more gracefully than top-k.
  * repetition penalty (CTRL-style) over a recent-context window.
  * [SEP] emission ends the document (that's what it means in packing).

    uv run python scripts/sample.py runs/val_ckpts/some.ckpt
    uv run python scripts/sample.py some.ckpt -p "In 1969 , the" -n 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

DEFAULT_PROMPTS = [
    "The water cycle begins when",
    "In this lesson , students will learn",
    "The Roman Empire was",
    "Scientists have discovered that",
]


def generate(model, encode, decode, sep_id, prompt, max_new=60,
             temperature=0.9, min_p=0.05, rep_penalty=1.2, rep_window=128):
    idx = torch.tensor(encode(prompt)).unsqueeze(0)
    for _ in range(max_new):
        logits = model(idx)[-1]

        # Repetition penalty over the recent context (CTRL-style: shrink
        # positive logits of recently seen tokens, grow negative ones).
        recent = torch.unique(idx[0, -rep_window:])
        picked = logits[recent]
        logits[recent] = torch.where(picked > 0,
                                     picked / rep_penalty,
                                     picked * rep_penalty)

        probs = F.softmax(logits / temperature, dim=-1)
        keep = probs >= min_p * probs.max()
        probs = torch.where(keep, probs, torch.zeros_like(probs))
        nxt = torch.multinomial(probs, 1)
        if int(nxt) == sep_id:
            break
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
    return decode(idx[0].tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('checkpoint')
    p.add_argument('--prompt', '-p', action='append', default=None,
                   help='repeatable; defaults to the standing eval prompts')
    p.add_argument('--max-new', '-n', type=int, default=60)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--min-p', type=float, default=0.05)
    p.add_argument('--rep-penalty', type=float, default=1.2)
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    import os
    os.environ.setdefault('TORCH_CPU_ONLY', '1')  # never fight a training run
    import text_data as td
    from summ_net import SummNet

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model = SummNet.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()
    encode = lambda s: td.tokenize(s, add_special_tokens=False)
    with torch.no_grad():
        for prompt in (args.prompt or DEFAULT_PROMPTS):
            out = generate(model, encode, td.decode, td.sep_token_id(), prompt,
                           max_new=args.max_new, temperature=args.temperature,
                           min_p=args.min_p, rep_penalty=args.rep_penalty)
            print(f"\n>>> {prompt}\n{out}")


if __name__ == '__main__':
    main()

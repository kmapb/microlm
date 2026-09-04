"""Pre-tokenize a dataset split into a packed uint16 token file.

One-time cost per (dataset, config, split); training then reads dense
windows from the memory-mapped pack with zero tokenization work.

    uv run python scripts/prepack.py HuggingFaceFW/fineweb-edu sample-10BT
    uv run python scripts/prepack.py Salesforce/wikitext wikitext-2-raw-v1 \
        --split train --split validation --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_data


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('dataset')
    p.add_argument('config', nargs='?', default=None)
    p.add_argument('--split', action='append', default=None,
                   help='repeatable; default: train')
    p.add_argument('--num-proc', type=int, default=16)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    for split in (args.split or ['train']):
        path = text_data.prepack(args.dataset, args.config, split,
                                 num_proc=args.num_proc, force=args.force)
        print(f"{split}: {path}")


if __name__ == '__main__':
    main()

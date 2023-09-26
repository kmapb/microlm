import torch
import summ_net as sn
import sys
import text_data

from util import dev

def syn_data(B, T):
    V = 256
    return torch.randint(0, V, (B, T)).to(dev())

def main(args):
    batchsz = args.batch_size
    torch.set_float32_matmul_precision('medium')

    for i in range(10, 100):
        inplen = int(1.4 ** i)
        mdl = sn.SummNet(text_data.vocabulary_size(),
                         dim = args.embedding_width,
                         fc_dim = args.fc_width,
                         height = args.wavenet_height,
                         max_length = inplen,
                         kernel_size = args.kernel_size).to(dev())

        print("Trying {}".format(inplen))
        optim = mdl.configure_optimizers()
        b = syn_data(batchsz, inplen)
        bd = { 'input_ids': b }

        optim.zero_grad()
        loss = mdl.training_step(bd, i)
        loss.backward()
        optim.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                        prog='test_input_sizes.py',
                        description='Expands max length until crash')
    parser.add_argument('--embedding-width', type=int, default=1024,
                        help='Embedding width')
    parser.add_argument('--fc-width', type=int, default=1024,
                        help='Classifier (FC) width')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--wavenet-height', type=int, default=13,
                        help='Wavenet height')
    parser.add_argument('--kernel-size', type=int, default=4,
            help = 'Kernel size')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to restore')
    args = parser.parse_args(sys.argv[1:])

    main(args)

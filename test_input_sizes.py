import torch
import summ_net as sn
import sys
import wandb

from util import dev

def syn_data(B, T):
    V = 256
    return torch.randint(0, V, (B, T)).to(dev())

def main(mdl):
    wandb.init(project='_test_microlm')
    batchsz = 1
    optim = mdl.configure_optimizers()
    torch.set_float32_matmul_precision('medium')

    for i in range(10, 100):
        inplen = int(1.2 ** i)
        print("Trying {}".format(inplen))
        b = syn_data(batchsz, inplen)
        bd = { 'input_ids': b }

        optim.zero_grad()
        loss = mdl.training_step(bd, i)
        loss.backward()
        optim.step()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        mdl = sn.SummNet().to(dev())
    else:
        mdl = sn.SummNet.load_from_checkpoint(sys.argv[1]).to(dev())
    main(mdl)

import torch
import text_data
import token_rnn
import pytorch_lightning as pl
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingDataModule,
    LanguageModelingTransformer,
)

from util import dev

def test_batch(model, xb, yb):
    with torch.no_grad():
        logits, loss = model(xb, targets=yb)
    return loss

def train_batch(model, opt, xb, yb):
  # sample a batch of data
  opt.zero_grad()
  
  # evaluate the loss
  logits, loss = model(xb, targets=yb)
  loss.backward()
  opt.step()
  return loss

def main(model, train, test, opt, batch_sz, nbatches, ctx_len):
    stepnum = 0
    test_losses = []
    train_losses = []
 
 
    K = 10 # Reporting stride
    train_gen = text_data.epoch_gen(train, batch_sz, ctx_len, max_samples=1024)
    test_gen = text_data.epoch_gen(test, batch_sz, ctx_len)
    print("Curriculum step: {} batches (batch size {}) length {}".format(nbatches, batch_sz, ctx_len))
    print("Example: {}".format(text_data.decode(model.generate())))
    for i in range(0, nbatches):
        xb, yb = next(train_gen, (False, False))
        if yb is False:
            print("Epoch done!")
            train_gen = text_data.epoch_gen(train, batch_sz, ctx_len, max_samples=1024)
            continue
        l = train_batch(model, opt, xb.to(dev()), yb.to(dev()))
        train_losses.append(l)
        if stepnum % K == 0:
            tx, ty = next(test_gen, (False, False))
            if ty is False:
                print("Exhausted test data?")
                test_loss = 0.0
            else:
                test_loss = test_batch(model, tx.to(dev()), ty.to(dev()))
            print("Step {}: train loss {} test loss {}".format(stepnum, torch.tensor(train_losses).mean(), test_loss))
            test_losses.append(test_loss)
        stepnum += 1
    print("done w/ mesobatch")
    return test_losses, train_losses
    

if __name__ == "__main__":
    CFG= {
        'model': 'conv_text',
        'dataset': 'the_pile',
        'dataset_cfg': 'all',
        'fname' : 'model-conv-text',
        'embed_width': 1024,
        'batch_size': 8,
        'context_width': 8192,
    }

    from conv_text import ConvText
    print("convo!!!")
    torch.set_float32_matmul_precision('medium')
    dataset = { 'name': 'wikitext', 'config': 'wikitext-2-raw-v1'}
    model = ConvText(text_data.vocabulary_size(), CFG['embed_width'], CFG['context_width']).to(dev())
    dm = LanguageModelingDataModule(batch_size = CFG['batch_size'],
                                    dataset_name = dataset['name'],
                                    dataset_config_name = dataset['config'],
                                    num_workers = 20,
                                    tokenizer=text_data._tokenizer())
    trainer = pl.Trainer(accelerator='auto', devices='auto', max_epochs=1)
    trainer.fit(model, dm)
    
    curriculum = [
        (128, 1024),
#        (256, 512),
 #       (512, 256),
        
    ]
    try:
        model = torch.load(CFG['fname'])
        print("restored model")
    except FileNotFoundError:
        model = None
    if model is None:   
        if CFG['model'] == 'conv_text':
            pass
        else:
            model = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
        print("christened new model")
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')
    dataset_name = 'the_pile'
    dataset_cfg = 'all'
    train = iter(text_data.load_dataset(dataset_name, dataset_cfg, 'train'))
    test = iter(text_data.load_dataset(dataset_name, dataset_cfg, 'test'))
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    # XXXkma: keep getting the same batch over and over again
    K = 20
    step = 0
    if True:
        eg = text_data.epoch_gen(train, CFG['batch_size'], CFG['context_width'])
        train_epoch = text_data.epoch_gen(train, CFG['batch_size'], CFG['context_width'], max_samples=1024)
        train_losses = []
        test_losses = []

        while True:
            step += 1
            if step % K == 1:
                te, tr = main(model, train, test, opt,
                            batch_sz=CFG['batch_size'],
                            nbatches=1, ctx_len=CFG['context_width'])
                print("test: {} train: {}".format(te, tr))
                torch.save(model.state_dict(), '{}-{}.pt'.format(CFG['fname'], step))
                train_losses += [t.cpu() for t in tr]
                torch.save({'test': test_losses, 'train': train_losses }, "training-log.pt")

            xb, yb = next(eg, (False, False))
            if yb is False:
                print("prefix down!")
                break
            xb = xb.to(dev())
            yb = yb.to(dev())
            print("batch: {}: {}".format(xb[0, :20], text_data.decode(xb[0, :20])))
            loss = train_batch(model, opt, xb, yb)
            test_losses.append(loss.cpu())
            print(loss)
            if loss < 1e-5:
                print("successfully overfit!")
                import sys; sys.exit(0)

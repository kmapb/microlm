import torch
import text_data
import token_rnn
from util import dev

def test_batch(model, xb, yb):
    with torch.no_grad():
        logits, loss = model(xb, targets=yb)
    return loss

def train_batch(model, opt, xb, yb):
  # sample a batch of data
  opt.zero_grad(set_to_none=True)
  
  # evaluate the loss
  logits, loss = model(xb, targets=yb)
  loss.backward()
  opt.step()
  return loss

def main(model, train, test, opt, batch_sz, nbatches, ctx_len):
    stepnum = 0
    for _ in range(nbatches):
        K = 10 # Reporting stride
        train_loss = 0.0
        train_gen = text_data.epoch_gen(train, batch_sz, ctx_len)
        test_gen = text_data.epoch_gen(test, batch_sz, ctx_len)
        print("Curriculum step: {} batches (batch size {}) length {}".format(nbatches, batch_sz, ctx_len))
        print("Example: {}".format(text_data.decode(model.generate())))
        for i in range(0, nbatches):
            xb, yb = next(train_gen, (False, False))
            if yb is False:
                print("Epoch done!")
                break
            train_loss += train_batch(model, opt, xb.to(dev()), yb.to(dev()))
            if stepnum % K == 0:
                tx, ty = next(test_gen, (False, False))
                if ty is False:
                    print("Exhausted test data?")
                    test_loss = 0.0
                else:
                    test_loss = test_batch(model, tx.to(dev()), ty.to(dev()))
                print("Step {}: train loss {} test loss {}".format(stepnum, train_loss / K, test_loss))
                train_loss = 0.0
            stepnum += 1
    

if __name__ == "__main__":
    CFG= {
        'model': 'conv_text',
        'dataset': 'the_pile',
        'dataset_cfg': 'all',
        'fname' : 'model-conv-text',
        'embed_width': 100,
        'context_width': 1024,
    }
    try:
        model = torch.load(CFG['fname'])
        print("restored model")
    except FileNotFoundError:
        model = None
    if model is None:   
        if CFG['model'] == 'conv_text':
            from conv_text import ConvText
            print("convo!!!")
            model = ConvText(text_data.vocabulary_size(), CFG['embed_width'], CFG['context_width']).to(dev())
        else:
            model = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
        print("christened new model")
    dataset_name = 'the_pile'
    dataset_cfg = 'all'
    train = iter(text_data.load_dataset(dataset_name, dataset_cfg, 'train'))
    test = iter(text_data.load_dataset(dataset_name, dataset_cfg, 'test'))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(1000):
        main(model, train, test, opt,
             batch_sz=32, nbatches=256, ctx_len=CFG['context_width'])
        torch.save(model, '{}-{}.pt'.format(CFG['fname'], i))
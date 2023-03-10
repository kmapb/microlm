import torch
import text_data
import token_rnn
from util import dev

def test_batch(model, xb, yb):
    total_loss = 0
    batch_size = xb.shape[0]
    block_size = xb.shape[1]
    assert(yb.shape[0] == batch_size)
    assert(yb.shape[1] == block_size)
    with torch.no_grad():
        hidden = None
        for i in range(block_size):
            xb_i = xb.select(1, i).reshape(batch_size, 1)
            yb_i = yb.select(1, i).reshape(batch_size, 1)
            logits, hidden, loss = model(xb_i, hidden, targets=yb_i)
            total_loss += loss
    return total_loss / block_size

def train_batch(model, opt, xb, yb):
  # sample a batch of data
  opt.zero_grad(set_to_none=True)
  total_loss = 0

  batch_size = xb.shape[0]
  block_size = xb.shape[1]
  assert(yb.shape[0] == batch_size)
  assert(yb.shape[1] == block_size)
  
  # evaluate the loss
  hidden = None
  for i in range(block_size):
      xb_i = xb.select(1, i).reshape(batch_size, 1)
      yb_i = yb.select(1, i).reshape(batch_size, 1)
      logits, hidden, loss = model(xb_i, hidden, targets=yb_i)
      total_loss += loss

  total_loss /= block_size
  total_loss.backward()
  opt.step()
  return total_loss

def main(model, train, test, opt, batch_sz, curriculum):
    stepnum = 0
    for nbatches, bptt in curriculum:
        K = 10 # Reporting stride
        train_loss = 0.0
        train_gen = text_data.epoch_gen(train, batch_sz, bptt)
        test_gen = text_data.epoch_gen(test, batch_sz, bptt)
        print("Curriculum step: {} batches (batch size {}) length {}".format(nbatches, batch_sz, bptt))
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
    try:
        model = torch.load('model.pt')
        print("restored model")
    except FileNotFoundError:
        model = None
    if model is None:   
        model = token_rnn.TokenRNNLM(text_data.vocabulary_size()).to(dev())
        print("christened new model")
    dataset_name = 'the_pile'
    dataset_cfg = 'all'
    train = text_data.load_dataset(dataset_name, dataset_cfg, 'train')
    test = text_data.load_dataset(dataset_name, dataset_cfg, 'test')
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    curriculum = [
        (1000, 10),
        (500, 30),
        (250, 100),
        (100, 300),
    #    (50, 1000)
    ]
    # curriculum = [ (64, 32), (100, 100)]
    train = iter(train)
    test = iter(test)
    for i in range(1000):
        main(model, train, test, opt,
             64,
             curriculum)
        torch.save(model, 'model-{}.pt'.format(i))
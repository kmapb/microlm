import torch
import text_data
import token_rnn
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
# from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule
from conv_text import ConvText, ReConvText

if __name__ == "__main__":
    import sys
    CFG= {
        'model': 'conv_text',
        'dataset': 'the_pile',
        'dataset_cfg': 'all',
        'fname' : 'model-conv-text',
        'embed_width': 348,
        'filter_height': 5,
    }
    CFG['context_width'] = (CFG['filter_height'] + 1) * CFG['embed_width']

    print("convo!!!")
    torch.set_float32_matmul_precision('medium')
    dataset = { 'name': 'wikitext', 'config': 'wikitext-103-raw-v1', 'streaming': True}
    # dataset = { 'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'streaming': False}
    # set = { 'name': 'the_pile', 'config': 'all', 'streaming': True}
    # model = ConvText(text_data.vocabulary_size(), CFG['embed_width'], CFG['context_width'])
    # XXXkma
    model = ReConvText(text_data.vocabulary_size(), CFG['filter_height'],
                       CFG['embed_width'], CFG['context_width'])
    
    pl.seed_everything(71177)
    dm = text_data.TextDataModule(dataset['name'], dataset['config'], streaming=dataset['streaming'])
    trainer = pl.Trainer(accelerator='auto',
                         devices='auto',
                         # max_time={'hours': 4},
                         overfit_batches=1,
                         log_every_n_steps=1,
                         limit_val_batches=10,
                         # limit_train_batches=1, # Overfit?
                         max_epochs=100,
                         # gradient_clip_val=0.5
                         )
    trainer.fit(model, dm)
    trainer.save_checkpoint('model.ckpt')
    # trainer.test(model, dm)

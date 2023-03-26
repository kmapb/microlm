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
        'embed_width': 512,
        'filter_height': 5,
    }
    CFG['context_width'] = (CFG['filter_height'] + 1) * CFG['embed_width']

    print("convo!!!")
    torch.set_float32_matmul_precision('medium')
    # dataset = { 'name': 'wikitext', 'config': 'wikitext-103-raw-v1', 'streaming': False}
    # dataset = { 'name': 'bookcorpus', 'config': 'plain_text', 'streaming': False}
    dataset = { 'name': 'wikitext', 'config': 'wikitext-2-raw-v1', 'streaming': False}
    # set = { 'name': 'the_pile', 'config': 'all', 'streaming': True}
    model = ReConvText(text_data.vocabulary_size(), CFG['filter_height'],
                       CFG['embed_width'], CFG['context_width'])
    
    pl.seed_everything(71177)
    dm = text_data.TextDataModule(dataset['name'], dataset['config'], streaming=dataset['streaming'])
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath="./val_ckpts",
        filename="train-ckpt-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(accelerator='auto',
                         devices='auto',
                         max_time={'hours': 4},
                         callbacks=[checkpoint_callback],
                         # overfit_batches=100,
                         # log_every_n_steps=1,
                         # limit_val_batches=10,
                         # limit_train_batches=1, # Overfit?
                         max_epochs=2,
                         # deterministic=True,
                         # gradient_clip_val=0.5
                         )


    trainer.fit(model, dm)
    trainer.save_checkpoint('model.ckpt')
    # trainer.test(model, dm)

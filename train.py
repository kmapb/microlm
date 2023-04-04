import torch
# import token_rnn
import pytorch_lightning as pl

import text_data
from conv_text import ReConvText

if __name__ == "__main__":
    import sys
    CFG= {
        'model': 'conv_text',
        # 'dataset': 'bookcorpus', 'dataset_cfg': 'plain_text',
        # 'dataset': 'the_pile', 'dataset_cfg': 'all',
        'dataset': 'wikitext', 'dataset_cfg': 'wikitext-2-v1', # quick test
        # 'dataset': 'wikitext', 'dataset_cfg': 'wikitext-103-v1',
        # 'dataset': 'c4', 'dataset_cfg': 'en',
        'dataset_pct': 0.1,
        'fname' : 'model-conv-text',
        'embed_width': 512,
        'filter_height': 5,
        'batch_size': 4,
    }
    
    # Compute the width of the first fully-connected layer; needs to be
    # big enough to accept the output of the convolutional layer.
    CFG['context_width'] = (CFG['filter_height'] + 1) * CFG['embed_width']

    # Allow the hardware to use mixed precision
    torch.set_float32_matmul_precision('medium')

    model = ReConvText(text_data.vocabulary_size(),
                       CFG['filter_height'],
                       CFG['embed_width'],
                       CFG['context_width'])
    
    pl.seed_everything(71177)
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath="./val_ckpts",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = pl.Trainer(accelerator='auto',
                         devices='auto',
                         max_time={'hours': 48},
                         callbacks=[checkpoint_callback],
                         val_check_interval=0.01,
                         #val_check_interval=0.25,
                         log_every_n_steps=100,
                         # overfit_batches=1000,
                         max_epochs=32,
                         )

    # dm = text_data.TextDataModule(CFG['dataset'], CFG['dataset_cfg'], streaming=False, pct=CFG['dataset_pct'])
    dm = text_data.BasicDataModule(CFG['dataset'], CFG['dataset_cfg'], pct=CFG['dataset_pct'], batch_size=CFG['batch_size'])
    trainer.fit(model, dm)
    trainer.save_checkpoint('model.ckpt')
    trainer.test(model, dm)

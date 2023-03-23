import torch
import text_data
import token_rnn
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule
from conv_text import ConvText, ReConvText

if __name__ == "__main__":
    import sys
    CFG= {
        'model': 'conv_text',
        'dataset': 'the_pile',
        'dataset_cfg': 'all',
        'fname' : 'model-conv-text',
        'embed_width': 1024,
        'filter_height': 5,
        'batch_size': 8,
    }
    CFG['context_width'] = (CFG['filter_height'] + 1) * CFG['embed_width']

    print("convo!!!")
    torch.set_float32_matmul_precision('medium')
    dataset = { 'name': 'wikitext', 'config': 'wikitext-103-raw-v1'}
    # model = ConvText(text_data.vocabulary_size(), CFG['embed_width'], CFG['context_width'])
    # XXXkma
    model = ReConvText(text_data.vocabulary_size(), CFG['filter_height'],
                       CFG['embed_width'], CFG['context_width'])
    dm = LanguageModelingDataModule(batch_size = CFG['batch_size'],
                                    dataset_name = dataset['name'],
                                    dataset_config_name = dataset['config'],
                                    num_workers = 20,
                                    tokenizer=text_data._tokenizer())
    trainer = pl.Trainer(accelerator='auto',
                         devices='auto',
                         max_epochs=10,
                         gradient_clip_val=0.5)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, dm, mode='power', steps_per_trial=10)
    trainer.fit(model, dm)
    trainer.test(ckpt_path='best')
    trainer.save_checkpoint('model.ckpt')
    sys.exit()

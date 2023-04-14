import torch
# import token_rnn
import pytorch_lightning as pl

import text_data
# from conv_text import ReConvText
from summ_net import SummNet

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(
                        prog='train.py',
                        description='Trains a model on a dataset',
                        epilog='May the odds be ever in your favor.')
    parser.add_argument('--dataset', type=str, default='bookcorpus',
                        help='Name of Huggingface dataset')
    parser.add_argument('--dataset-cfg', type=str, default=None,
                        help='Config of Huggingface dataset')
    parser.add_argument('--max-hours', type=int, default=48,
                        help='Maximum number of hours to train')
    parser.add_argument('--max-epochs', type=int, default=2,
                        help='Maximum number of epochs to train')
    parser.add_argument('--checkpoint-restore', '-c', type=str, default=None,
                        help='Checkpoint to restore')
    parser.add_argument('--embedding-width', type=int, default=1024,
                        help='Embedding width')
    parser.add_argument('--fc-width', type=int, default=512,
                        help='Classifier (FC) width')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum length of input')
    parser.add_argument('--wavenet-height', type=int, default=13,
                        help='Wavenet height')
    parser.add_argument('--random-seed', type=int, default=22707,
                        help='Random seed')
    args = parser.parse_args(argv)
    
    # dataset: 'bookcorpus'
    # dataset: 'the_pile', dataset_cfg: 'all'
    # dataset: 'wikitext', dataset_cfg: 'wikitext-2-v1', # quick test
    # dataset: 'wikitext', dataset_cfg: 'wikitext-103-v1',
    # dataset: 'c4', 'dataset_cfg': 'en',
        
    # Allow the hardware to use mixed precision
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(args.random_seed)
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath="./val_ckpts",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}",
    )
    model = None
    if args.checkpoint_restore:
        print("restoring from checkpoint {}".format(args.checkpoint_restore))
        model = SummNet.load_from_checkpoint(args.checkpoint_restore)
    else:
        print("creating new model")
        model = SummNet(text_data.vocabulary_size(),
                        dim = args.embedding_width,
                        fc_dim = args.fc_width,
                        height = args.wavenet_height,
                        max_length = args.max_length)
        
    trainer = pl.Trainer(accelerator='auto',
                         devices='auto',
                         max_time={'hours': args.max_hours},
                         callbacks=[checkpoint_callback],
                         val_check_interval=0.001,
                         log_every_n_steps=100,
                         limit_val_batches=0.001,
                         max_epochs=args.max_epochs,
                         )

    dm = text_data.BasicDataModule(args.dataset, args.dataset_cfg, max_length=args.max_length,
                                   batch_size=args.batch_size)
    trainer.fit(model, dm)
    trainer.save_checkpoint('full-run.ckpt')
    trainer.test(model, dm)
    
if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

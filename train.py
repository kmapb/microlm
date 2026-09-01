import paths  # noqa: F401  -- sets cache locations; must precede datasets/transformers

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks as PLCB
from pytorch_lightning import loggers as PLLG

import text_data
# from conv_text import ReConvText
from summ_net import SummNet
from typing import List, cast


# The shared tracking server; auth is handled at the network perimeter.
MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', 'https://mlflow.pbd.vc')


class FailSoftMLFlowLogger(PLLG.MLFlowLogger):
    """A metrics-DB blip must not kill a training run (a ~3-minute managed-
    Postgres failover took down 6 GPU-hours on 2026-08-29): drop the points
    and keep training instead of raising through Lightning."""

    def log_metrics(self, metrics, step=None):
        try:
            super().log_metrics(metrics, step)
        except Exception as e:
            print(f"[mlflow] dropped metrics at step {step}: {e}")

    def log_hyperparams(self, params):
        try:
            super().log_hyperparams(params)
        except Exception as e:
            print(f"[mlflow] dropped hparams: {e}")

    def finalize(self, status):
        try:
            super().finalize(status)
        except Exception as e:
            print(f"[mlflow] finalize failed: {e}")


def make_logger(kind: str, run_name: str = None, run_id: str = None):
    """mlflow (the shared server) is the default; the others are for offline
    boxes and quick local runs. wandb needs a login -- ask for it explicitly."""
    if kind == 'none':
        return False
    if kind == 'mlflow':
        # run_id resumes logging into an existing run (crash recovery);
        # otherwise a fresh run is created under run_name.
        return FailSoftMLFlowLogger(experiment_name="microlm",
                                    run_name=run_name,
                                    run_id=run_id,
                                    tracking_uri=MLFLOW_URI)
    if kind == 'tensorboard':
        return PLLG.TensorBoardLogger(save_dir=str(paths.LOG_DIR), name="microlm")
    if kind == 'csv':
        return PLLG.CSVLogger(save_dir=str(paths.LOG_DIR), name="microlm")
    if kind == 'wandb':
        return PLLG.WandbLogger(name="microlm", save_dir=str(paths.LOG_DIR), log_model=True)
    raise ValueError(f"unknown logger {kind!r}")


def main(argv: List[str]):
    import argparse
    parser = argparse.ArgumentParser(
                        prog='train.py',
                        description='Trains a model on a dataset',
                        epilog='May the odds be ever in your favor.')
    parser.add_argument('--dataset', type=str, default='Salesforce/wikitext',
                        help='Name of Huggingface dataset')
    parser.add_argument('--dataset-cfg', type=str, default='wikitext-103-raw-v1',
                        help='Config of Huggingface dataset')
    parser.add_argument('--streaming', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Stream the dataset instead of downloading it first')
    parser.add_argument('--max-hours', type=float, default=0.5,
                        help='Maximum number of hours to train')
    parser.add_argument('--max-epochs', type=int, default=2,
                        help='Maximum number of epochs to train')
    parser.add_argument('--checkpoint-restore', '-c', type=str, default=None,
                        help='Checkpoint to restore')
    parser.add_argument('--embedding-width', type=int, default=1024,
                        help='Embedding width')
    parser.add_argument('--fc-width', type=int, default=1024,
                        help='Fully-conncted layer width')
    parser.add_argument('--kernel-size', type=int, default=3,
                        help='Receptive field for convolutions')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max-length', type=int, default=4096,
                        help='Maximum length of input')
    parser.add_argument('--wavenet-height', type=int, default=None,
                        help='Wavenet height (default: smallest stack whose '
                             'receptive field covers --max-length)')
    parser.add_argument('--random-seed', type=int, default=22707,
                        help='Random seed')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Peak learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Linear LR warmup steps')
    parser.add_argument('--lr-decay-steps', type=int, default=100_000,
                        help='Steps over which LR cosine-decays to a 10%% floor')
    parser.add_argument('--arch', type=str, default='v2',
                        choices=['v1', 'v2', 'v3', 'v4', 'v4m', 't1'],
                        help='v1 = original blocks; v2 = GLU gating + skip '
                             'aggregation + tied embeddings; v3 = v2 without '
                             'positional embeddings; v4 = dilated QKV '
                             'attention on the wavenet tree; v4m = v4 with '
                             'transformer-style MLP sub-layers; t1 = GPT-2-'
                             'style transformer baseline (height = layers)')
    parser.add_argument('--window', type=int, default=8,
                        help='v4: attention candidates per level')
    parser.add_argument('--cycles', type=int, default=3,
                        help='v4: repeats of the dilation ladder')
    parser.add_argument('--logger', type=str, default='mlflow',
                        choices=['mlflow', 'tensorboard', 'csv', 'wandb', 'none'],
                        help='Where to send metrics')
    parser.add_argument('--limit-train-batches', type=int, default=None,
                        help='Cap training batches per epoch (handy for smoke tests)')
    parser.add_argument('--val-check-interval', type=int, default=1000,
                        help='Training batches between validation passes')
    parser.add_argument('--test-only', action='store_true',
                        help='Just test the checkpoint')
    parser.add_argument('--mlflow-run-id', type=str, default=None,
                        help='Resume logging into this existing MLflow run '
                             '(pairs with --checkpoint-restore)')
    args = parser.parse_args(argv)

    # dataset: 'Salesforce/wikitext', dataset_cfg: 'wikitext-2-raw-v1', # quick test
    # dataset: 'Salesforce/wikitext', dataset_cfg: 'wikitext-103-raw-v1',
    # dataset: 'HuggingFaceFW/fineweb-edu', dataset_cfg: 'sample-10BT',
    # dataset: 'allenai/c4', dataset_cfg: 'en',

    paths.ensure_dirs()
    print("microlm paths:")
    print(paths.describe())

    # Lightning refuses to start if it would never reach a validation pass, which
    # is exactly what happens on a short --limit-train-batches run.
    val_check_interval = args.val_check_interval
    if args.limit_train_batches is not None:
        val_check_interval = min(val_check_interval, args.limit_train_batches)

    # Allow the hardware to use mixed precision
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(args.random_seed)
    # saves top-K checkpoints based on "val_loss" metric
    arch_tag = args.arch
    if args.arch.startswith('v4'):
        # The n-ariness sweep needs distinguishable runs/checkpoints.
        arch_tag = "{}w{}x{}".format(args.arch, args.window, args.cycles)
    filename_template = "ckpt-{}-k{}-d{}".format(
        arch_tag, args.kernel_size, args.embedding_width)
    checkpoint_callback = PLCB.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath=str(paths.CHECKPOINT_DIR),
        filename=filename_template + "-{val_loss:.2f}"
    )
    model = None
    if args.checkpoint_restore:
        print("restoring from checkpoint {}".format(args.checkpoint_restore))
        model = cast(SummNet, SummNet.load_from_checkpoint(args.checkpoint_restore))
        args.max_length = model.max_length
    else:
        print("creating new model")
        model = SummNet(text_data.vocabulary_size(),
                        dim=args.embedding_width,
                        fc_dim=args.fc_width,
                        height=args.wavenet_height,
                        max_length=args.max_length,
                        kernel_size=args.kernel_size,
                        pad_token_id=text_data.pad_token_id(),
                        lr=args.lr,
                        warmup_steps=args.warmup_steps,
                        lr_decay_steps=args.lr_decay_steps,
                        arch=args.arch,
                        window=args.window,
                        cycles=args.cycles)

    trainer = pl.Trainer(accelerator='auto',
                         precision='16-mixed' if torch.cuda.is_available() else '32-true',
                         devices=1,
                         max_time={'hours': args.max_hours},
                         gradient_clip_val=1.0,
                         callbacks=[checkpoint_callback],
                         val_check_interval=val_check_interval,
                         log_every_n_steps=100,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=337,
                         limit_test_batches=8000,
                         max_epochs=args.max_epochs,
                         logger=make_logger(args.logger,
                                            run_name="{}-{}-k{}-d{}".format(
                                                args.dataset.split('/')[-1],
                                                arch_tag,
                                                args.kernel_size,
                                                args.embedding_width),
                                            run_id=args.mlflow_run_id),
                         )

    stream_factory = text_data.StreamingTextDataModule if args.streaming else text_data.BasicDataModule
    dm = stream_factory(args.dataset, args.dataset_cfg,
                        max_length=args.max_length,
                        batch_size=args.batch_size)
    if not args.test_only:
        trainer.fit(model, dm, ckpt_path=args.checkpoint_restore)
        final = paths.OUTPUT_ROOT / 'full-run-d{}-h{}.ckpt'.format(
            args.embedding_width, args.kernel_size)
        trainer.save_checkpoint(str(final))
        print("wrote {}".format(final))
    trainer.test(model, dm)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

"""Replay a TensorBoard run's scalars into the shared MLflow server.

For runs that logged locally (--logger tensorboard, or anything predating
mlflow support): reads every scalar series out of the tfevents files, keeps
the original steps and wall-clock timestamps, and pushes them to MLflow as
one run, with the hparams.yaml attached as params.

Safe to re-run: --replace deletes the previous same-named run first, so the
pattern for an in-progress training run is "backfill now to watch, backfill
again with --replace when it finishes".

    uv run python scripts/backfill_mlflow.py runs/logs/microlm/version_3 \
        --run-name wt103-d1024 --replace
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# MLflow rejects batches over 1000 entities.
BATCH = 900


def thin(events, cap):
    """Evenly downsample to `cap` points, always keeping the last one."""
    if len(events) <= cap:
        return events
    stride = len(events) / cap
    picked = [events[int(i * stride)] for i in range(cap)]
    picked[-1] = events[-1]
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('logdir', help='TensorBoard run dir, e.g. runs/logs/microlm/version_3')
    p.add_argument('--run-name', required=True)
    p.add_argument('--experiment', default='microlm')
    p.add_argument('--tracking-uri',
                   default=os.environ.get('MLFLOW_TRACKING_URI', 'https://mlflow.pbd.vc'))
    p.add_argument('--replace', action='store_true',
                   help='delete an existing run with the same name first')
    p.add_argument('--max-points', type=int, default=2000,
                   help='per-series downsampling cap (train_tokens logs every step)')
    args = p.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    exp = client.get_experiment_by_name(args.experiment)
    exp_id = exp.experiment_id if exp else client.create_experiment(args.experiment)

    for r in client.search_runs([exp_id], f"attributes.run_name = '{args.run_name}'"):
        if not args.replace:
            raise SystemExit(
                f"run {args.run_name!r} already exists ({r.info.run_id}); use --replace")
        client.delete_run(r.info.run_id)
        print(f"deleted prior run {r.info.run_id}")

    run_id = client.create_run(exp_id, run_name=args.run_name).info.run_id

    hparams = Path(args.logdir) / 'hparams.yaml'
    if hparams.exists():
        params = yaml.safe_load(hparams.read_text()) or {}
        client.log_batch(run_id, params=[
            Param(str(k), str(v)[:500]) for k, v in params.items()])

    # size_guidance 0 = keep every event instead of the default reservoir sample.
    ea = EventAccumulator(args.logdir, size_guidance={'scalars': 0})
    ea.Reload()
    total = 0
    for tag in ea.Tags()['scalars']:
        events = thin(ea.Scalars(tag), args.max_points)
        steps = [e.step for e in events]
        if len(events) > 1 and len(set(steps)) == 1:
            # Series logged without a step (train_tokens) all land on step 0;
            # index them so they plot as a series rather than a point.
            steps = list(range(len(events)))
        metrics = [Metric(tag, e.value, int(e.wall_time * 1000), s)
                   for e, s in zip(events, steps)]
        for i in range(0, len(metrics), BATCH):
            client.log_batch(run_id, metrics=metrics[i:i + BATCH])
        total += len(metrics)
        print(f"  {tag}: {len(metrics)} points")

    client.set_terminated(run_id)
    print(f"backfilled {total} points to {args.tracking_uri} "
          f"experiment {args.experiment!r} run {args.run_name!r} ({run_id})")


if __name__ == '__main__':
    main()

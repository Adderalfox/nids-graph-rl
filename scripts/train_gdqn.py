"""Experiment B: GAT-based dueling Double DQN (graph + RL combined).

Examples:
    # smoke
    uv run python scripts/train_gdqn.py --source ton --seeds 0 --epochs 1 --max-updates 3000 \
        --val-stride 2 --eval-stride 10 --drop-groups host_fingerprint
    # reported (5 seeds) and the no-message-passing control under identical RL
    uv run python scripts/train_gdqn.py --source ton --seeds 0 1 2 3 4 \
        --drop-groups host_fingerprint
    uv run python scripts/train_gdqn.py --source ton --seeds 0 1 2 3 4 --layers 0 --tag nomp \
        --drop-groups host_fingerprint
"""

from __future__ import annotations

import argparse
import json

import yaml

from nidsrl.config import load_config
from nidsrl.experiments.graph_dqn import run_graph_dqn
from nidsrl.experiments.rl_common import format_pair
from nidsrl.models.features import FEATURE_GROUPS
from nidsrl.rl.trainer import TrainConfig
from nidsrl.utils.runs import create_run_dir

GROUP_CHOICES = [*sorted(FEATURE_GROUPS), "flowcount"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["unsw", "cic", "ton"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--drop-groups", nargs="*", default=[], choices=GROUP_CHOICES)
    ap.add_argument("--keep-groups", nargs="*", default=[], choices=GROUP_CHOICES)
    ap.add_argument("--tag", default="")
    d = TrainConfig(epochs=6, learn_start=3_000, buffer_capacity=150_000, updates_per_window=1)
    ap.add_argument("--layers", type=int, default=2, help="0 = no message passing")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--windows-per-update", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--updates-per-window", type=int, default=d.updates_per_window)
    ap.add_argument("--learn-start", type=int, default=d.learn_start)
    ap.add_argument("--buffer", type=int, default=d.buffer_capacity, help="transitions kept")
    ap.add_argument("--gamma", type=float, default=d.gamma)
    ap.add_argument("--eps-decay-windows", type=int, default=d.eps_decay_windows)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--fp-cost", type=float, default=1.0)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--val-stride", type=int, default=1)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    tc = TrainConfig(
        epochs=args.epochs, max_updates=args.max_updates,
        updates_per_window=args.updates_per_window,
        learn_start=args.learn_start, buffer_capacity=args.buffer, gamma=args.gamma,
        eps_decay_windows=args.eps_decay_windows, patience=args.patience, double=True,
    )  # fmt: skip
    for seed in args.seeds:
        cfg = load_config("configs/default.yaml", [f"project.seed={seed}"])
        out = run_graph_dqn(
            cfg.paths.data_processed, args.source, seed, layers=args.layers, hidden=args.hidden,
            heads=args.heads, lr=args.lr, tau=args.tau, windows_per_update=args.windows_per_update,
            train_cfg=tc, drop_groups=tuple(args.drop_groups), keep_groups=tuple(args.keep_groups),
            fp_cost=args.fp_cost, fpr_target=args.fpr_target, val_stride=args.val_stride,
            eval_stride=args.eval_stride, targets=args.targets, tag=args.tag,
            device=cfg.project.device,
        )  # fmt: skip
        for key, result in out.items():
            m = result["meta"]
            label = "-".join(x for x in (m["model"], m["variant"], args.source) if x)
            run_dir = create_run_dir(cfg, tag=f"rl-{label}")
            (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (run_dir / "experiment.yaml").write_text(
                yaml.safe_dump(json.loads(json.dumps(m)), sort_keys=False), encoding="utf-8"
            )
            print(f"[{key}]")
            for p in result["pairs"]:
                print(format_pair(p))
            print(f"  saved: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

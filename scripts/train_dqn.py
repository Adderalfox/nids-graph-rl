"""Experiment A: train an MLP DQN-family agent on one source dataset, evaluate on all targets.

Examples:
    # quick smoke run (few updates, thinned evaluation)
    uv run python scripts/train_dqn.py --source unsw --seeds 0 --epochs 1 --max-updates 2000 \
        --val-stride 2 --eval-stride 10
    # reported run (main feature policy: drop the host_fingerprint shortcut group)
    uv run python scripts/train_dqn.py --source ton --seeds 0 1 2 3 4 --drop-groups host_fingerprint
    # controls: all features / contextual bandit (gamma 0) / plain DQN
    uv run python scripts/train_dqn.py --source ton --seeds 0 --tag allfeat
    uv run python scripts/train_dqn.py --source ton --seeds 0 --drop-groups host_fingerprint \
        --gamma 0 --tag gamma0
"""

from __future__ import annotations

import argparse
import json

import yaml

from nidsrl.config import load_config
from nidsrl.experiments.dqn import VARIANTS, run_dqn
from nidsrl.models.features import FEATURE_GROUPS
from nidsrl.rl.trainer import TrainConfig
from nidsrl.utils.runs import create_run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["unsw", "cic", "ton"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="d3qn")
    ap.add_argument("--drop-groups", nargs="*", default=[], choices=sorted(FEATURE_GROUPS))
    ap.add_argument("--keep-groups", nargs="*", default=[], choices=sorted(FEATURE_GROUPS))
    ap.add_argument("--tag", default="")
    d = TrainConfig()
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--max-updates", type=int, default=d.max_updates)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--updates-per-window", type=int, default=d.updates_per_window)
    ap.add_argument("--learn-start", type=int, default=d.learn_start)
    ap.add_argument("--buffer", type=int, default=d.buffer_capacity)
    ap.add_argument("--gamma", type=float, default=d.gamma)
    ap.add_argument("--eps-decay-windows", type=int, default=d.eps_decay_windows)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--inspect-windows", type=int, default=1)
    ap.add_argument("--block-windows", type=int, default=6)
    ap.add_argument("--val-stride", type=int, default=1, help="use every k-th val run")
    ap.add_argument("--eval-stride", type=int, default=1, help="1 for reported numbers")
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    tc = TrainConfig(
        epochs=args.epochs, max_updates=args.max_updates, batch_size=args.batch_size,
        updates_per_window=args.updates_per_window, learn_start=args.learn_start,
        buffer_capacity=args.buffer, gamma=args.gamma,
        eps_decay_windows=args.eps_decay_windows, patience=args.patience,
    )  # fmt: skip
    for seed in args.seeds:
        cfg = load_config("configs/default.yaml", [f"project.seed={seed}"])
        result = run_dqn(
            cfg.paths.data_processed, args.source, seed, variant=args.variant,
            drop_groups=tuple(args.drop_groups), keep_groups=tuple(args.keep_groups),
            train_cfg=tc, lr=args.lr, tau=args.tau, inspect_windows=args.inspect_windows,
            block_windows=args.block_windows, val_stride=args.val_stride,
            eval_stride=args.eval_stride, targets=args.targets, tag=args.tag,
            device=cfg.project.device,
            ckpt_path=None,
        )  # fmt: skip
        m = result["meta"]
        label = "-".join(x for x in (m["model"], m["variant"], args.source) if x)
        run_dir = create_run_dir(cfg, tag=f"rl-{label}")
        (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (run_dir / "experiment.yaml").write_text(
            yaml.safe_dump(json.loads(json.dumps(m)), sort_keys=False), encoding="utf-8"
        )
        for p in result["pairs"]:
            b, r, rs = p["binary"], p["ranking"], p["response"]
            print(
                f"  {p['source']}->{p['target']:<4} macroF1={b['macro_f1']:.3f} "
                f"TPR={b['tpr']:.3f} FPR={b['fpr']:.4f} AUC={r['roc_auc']:.3f} "
                f"prevented={rs['prevented_attack_share']:.3f} "
                f"collateral={rs['collateral_benign_share']:.3f}"
                if None not in (b["macro_f1"], b["tpr"], b["fpr"], r["roc_auc"])
                and None not in (rs["prevented_attack_share"], rs["collateral_benign_share"])
                else f"  {p['source']}->{p['target']:<4} (see results.json)"
            )
        print(f"  saved: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment G: supervised GAT edge classifier on window graphs (no RL).

Examples:
    # smoke
    uv run python scripts/train_gat.py --source ton --seeds 0 --epochs 2 --eval-stride 10 \
        --drop-groups host_fingerprint
    # ablation without message passing (edge MLP)
    uv run python scripts/train_gat.py --source ton --seeds 0 --layers 0 --tag nomp \
        --drop-groups host_fingerprint
"""

from __future__ import annotations

import argparse
import json

from nidsrl.config import load_config
from nidsrl.experiments.gat_supervised import run_gat_supervised
from nidsrl.models.features import FEATURE_GROUPS
from nidsrl.utils.runs import create_run_dir

GROUP_CHOICES = [*sorted(FEATURE_GROUPS), "flowcount"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["unsw", "cic", "ton"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--drop-groups", nargs="*", default=[], choices=GROUP_CHOICES)
    ap.add_argument("--keep-groups", nargs="*", default=[], choices=GROUP_CHOICES)
    ap.add_argument("--tag", default="")
    ap.add_argument("--layers", type=int, default=2, help="0 = no message passing")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--windows-per-batch", type=int, default=24)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--fpr-target", type=float, default=0.01)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--targets", nargs="*", default=None)
    args = ap.parse_args()

    for seed in args.seeds:
        cfg = load_config("configs/default.yaml", [f"project.seed={seed}"])
        out = run_gat_supervised(
            cfg.paths.data_processed, args.source, seed, layers=args.layers,
            hidden=args.hidden, heads=args.heads, lr=args.lr, epochs=args.epochs,
            windows_per_batch=args.windows_per_batch, patience=args.patience,
            drop_groups=tuple(args.drop_groups), keep_groups=tuple(args.keep_groups),
            fpr_target=args.fpr_target, eval_stride=args.eval_stride, targets=args.targets,
            tag=args.tag, device=cfg.project.device,
        )  # fmt: skip
        for name, result in out.items():
            m = result["meta"]
            label = "-".join(x for x in (m["model"], m["variant"], args.source) if x)
            run_dir = create_run_dir(cfg, tag=f"rl-{label}")
            (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[{name}]")
            for p in result["pairs"]:
                b, r = p["binary"], p["ranking"]
                if None in (b["macro_f1"], b["tpr"], b["fpr"], r["roc_auc"]):
                    print(f"  {p['source']}->{p['target']:<4} (see results.json)")
                else:
                    print(
                        f"  {p['source']}->{p['target']:<4} macroF1={b['macro_f1']:.3f} "
                        f"TPR={b['tpr']:.3f} FPR={b['fpr']:.4f} AUC={r['roc_auc']:.3f}"
                    )
            print(f"  saved: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

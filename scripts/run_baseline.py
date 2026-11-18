"""Train supervised baselines and evaluate them on every target dataset.

Examples:
    # quick development pass (thinned evaluation, one seed)
    uv run python scripts/run_baseline.py --models hgb --sources ton --seeds 0 --row-stride 10
    # full run for reported numbers
    uv run python scripts/run_baseline.py --models logreg rf hgb \\
        --sources unsw cic ton --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.experiments.baseline import run_baseline
from nidsrl.models.baselines import BASELINES
from nidsrl.utils.runs import create_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["hgb"], choices=BASELINES)
    parser.add_argument("--sources", nargs="+", default=["unsw", "cic", "ton"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-cap", type=int, default=150_000)
    parser.add_argument("--val-cap", type=int, default=50_000)
    parser.add_argument("--fpr-target", type=float, default=0.01)
    parser.add_argument("--row-stride", type=int, default=1)
    args = parser.parse_args()

    for model in args.models:
        for source in args.sources:
            for seed in args.seeds:
                cfg = load_config("configs/default.yaml", [f"project.seed={seed}"])
                root = Path(cfg.paths.data_processed)
                result = run_baseline(
                    root,
                    source,
                    model,
                    seed,
                    train_cap=args.train_cap,
                    val_cap=args.val_cap,
                    fpr_target=args.fpr_target,
                    row_stride=args.row_stride,
                )
                run_dir = create_run_dir(cfg, tag=f"baseline-{model}-{source}")
                (run_dir / "results.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                for p in result["pairs"]:
                    b, r = p["binary"], p["ranking"]
                    print(
                        f"  {p['source']}->{p['target']:<4} macroF1={b['macro_f1']:.3f} "
                        f"TPR={b['tpr']:.3f} FPR={b['fpr']:.4f} AUC={r['roc_auc']:.3f}"
                        if r and r["roc_auc"] is not None and b["macro_f1"] is not None
                        else f"  {p['source']}->{p['target']:<4} (see results.json)"
                    )
                print(f"  saved: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

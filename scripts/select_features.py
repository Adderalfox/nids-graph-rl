"""Nested source-only feature-policy selection (resumable).

Example (about 1-2 hours for unsw/cic/ton):
    uv run python scripts/select_features.py --sources unsw cic ton --eval-stride 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.experiments.feature_selection import run_selection


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=["unsw", "cic", "ton"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-stride", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config("configs/default.yaml", [f"project.seed={args.seed}"])
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "feature_selection_raw.json"
    done = json.loads(raw_path.read_text()) if raw_path.exists() else {}

    def save(key, pairs):
        done[key] = pairs
        raw_path.write_text(json.dumps(done, indent=2), encoding="utf-8")

    res = run_selection(
        cfg.paths.data_processed, args.sources, args.seed,
        eval_stride=args.eval_stride, done=done, on_result=save,
    )  # fmt: skip
    (out_dir / "feature_selection.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    cands = list(next(iter(res["per_heldout"].values()))["scores"])
    heads = " | ".join(f"w/o {h}" for h in res["per_heldout"])
    lines = [f"| candidate | {heads} | all |"]
    lines.append("|" + "---|" * (len(res["per_heldout"]) + 2))
    for c in cands:
        row = [f"{res['per_heldout'][h]['scores'][c]:.3f}" for h in res["per_heldout"]]
        lines.append(f"| {c} | " + " | ".join(row) + f" | {res['global']['scores'][c]:.3f} |")
    text = "\n".join(lines)
    (out_dir / "feature_selection.md").write_text(text + "\n", encoding="utf-8")
    print("Mean source-source ROC-AUC per candidate (selection score):\n" + text)
    for h, v in res["per_heldout"].items():
        print(f"held-out {h}: pick = {v['pick']} (selected on {v['selected_on']})")
    print(f"global pick (main configuration): {res['global']['pick']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

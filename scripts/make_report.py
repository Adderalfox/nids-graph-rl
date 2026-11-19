"""Aggregate baseline runs into source-by-target matrices.

Example:
    uv run python scripts/make_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.eval.report import collect_results, matrix_markdown, summarize

METRICS = ("macro_f1", "roc_auc", "tpr_at_fpr_1pct", "fpr", "unseen_detection_rate")
RL_METRICS = (*METRICS, "prevented", "collateral")
OUT = {"baseline": "baselines.md", "rl_env": "rl_env.md"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(OUT), default="baseline")
    args = parser.parse_args()
    cfg = load_config("configs/default.yaml")
    df = collect_results(cfg.paths.runs, kind=args.kind)
    if df.empty:
        print(f"No '{args.kind}' results found in {cfg.paths.runs}.")
        return 1
    summary = summarize(df)
    out = [f"# {args.kind} results (mean ± std over seeds; diagonal = in-dataset test rows)\n"]
    for model in sorted(summary["model"].unique()):
        for metric in RL_METRICS if args.kind == "rl_env" else METRICS:
            out.append(f"## {model}: {metric}\n")
            out.append(matrix_markdown(summary, model, metric) + "\n")
    text = "\n".join(out)
    out_path = Path("experiments/results") / OUT[args.kind]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

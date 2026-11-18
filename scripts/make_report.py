"""Aggregate baseline runs into source-by-target matrices.

Example:
    uv run python scripts/make_report.py
"""

from __future__ import annotations

from pathlib import Path

from nidsrl.config import load_config
from nidsrl.eval.report import collect_results, matrix_markdown, summarize

METRICS = ("macro_f1", "roc_auc", "tpr_at_fpr_1pct", "fpr", "unseen_detection_rate")


def main() -> int:
    cfg = load_config("configs/default.yaml")
    df = collect_results(cfg.paths.runs, kind="baseline")
    if df.empty:
        print("No baseline results found. Run scripts/run_baseline.py first.")
        return 1
    summary = summarize(df)
    out = ["# Baseline results (mean ± std over seeds; diagonal = in-dataset test rows)\n"]
    for model in sorted(summary["model"].unique()):
        for metric in METRICS:
            out.append(f"## {model}: {metric}\n")
            out.append(matrix_markdown(summary, model, metric) + "\n")
    text = "\n".join(out)
    out_path = Path("experiments/results/baselines.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

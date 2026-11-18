"""Aggregate run results into source-by-target matrices (mean +- std over seeds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "macro_f1": ("binary", "macro_f1"),
    "tpr": ("binary", "tpr"),
    "fpr": ("binary", "fpr"),
    "roc_auc": ("ranking", "roc_auc"),
    "tpr_at_fpr_1pct": ("ranking", "tpr_at_fpr_0.01"),
    "tpr_at_fpr_0.1pct": ("ranking", "tpr_at_fpr_0.001"),
    "pr_auc": ("ranking", "pr_auc"),
    "shared_macro_recall": ("groups", "shared_macro_recall"),
    "unseen_detection_rate": ("groups", "unseen_detection_rate"),
}


def _dig(d: dict[str, Any], path: tuple[str, ...]) -> float | None:
    for key in path:
        if d is None:
            return None
        d = d.get(key)
    return d


def collect_results(runs_dir: str | Path, kind: str = "baseline") -> pd.DataFrame:
    """One row per (run, pair) with the metrics in METRIC_PATHS."""
    rows = []
    for path in sorted(Path(runs_dir).glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if meta.get("kind") != kind:
            continue
        for pair in data["pairs"]:
            row = {
                "model": meta["model"],
                "source": pair["source"],
                "target": pair["target"],
                "kind": pair["kind"],
                "seed": meta["seed"],
                "n_flows": pair["n_flows"],
            }
            for name, mpath in METRIC_PATHS.items():
                row[name] = _dig(pair, mpath)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and seed count per (model, source, target)."""
    if df.empty:
        return df
    metrics = list(METRIC_PATHS)
    grouped = df.groupby(["model", "source", "target"], sort=True)
    mean = grouped[metrics].mean()
    std = grouped[metrics].std(ddof=0)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["n_seeds"] = grouped["seed"].nunique()
    return out.reset_index()


def _fmt(mean: float, std: float, n: int) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "-"
    return f"{mean:.3f}±{std:.3f}" if n > 1 else f"{mean:.3f}"


def matrix_markdown(summary: pd.DataFrame, model: str, metric: str) -> str:
    """Source-by-target markdown table for one model and metric."""
    sub = summary[summary["model"] == model]
    if sub.empty:
        return f"(no results for {model})"
    sources = sorted(sub["source"].unique())
    targets = sorted(sub["target"].unique())
    lines = [
        f"| source \\ target | {' | '.join(targets)} |",
        "|---|" + "---|" * len(targets),
    ]
    for s in sources:
        cells = []
        for t in targets:
            r = sub[(sub["source"] == s) & (sub["target"] == t)]
            if r.empty:
                cells.append("-")
            else:
                row = r.iloc[0]
                cells.append(_fmt(row[f"{metric}_mean"], row[f"{metric}_std"], int(row["n_seeds"])))
        lines.append(f"| {s} | {' | '.join(cells)} |")
    return "\n".join(lines)

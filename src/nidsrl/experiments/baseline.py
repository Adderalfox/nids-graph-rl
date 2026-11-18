"""Run one supervised baseline for one source dataset and evaluate it on every target.

Protocol (docs/evaluation.md): fit on a category-capped sample of source train,
choose the decision threshold on a source validation sample (FPR budget), then
evaluate on every protocol pair of this source with the source scaler.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nidsrl.data.sampling import sample_split
from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.eval.harness import evaluate_pair
from nidsrl.eval.metrics import evaluate_arrays, threshold_at_fpr
from nidsrl.models.baselines import make_baseline
from nidsrl.models.features import kept_feature_indices
from nidsrl.utils.seed import seed_everything


def ablation_label(drop_groups: tuple[str, ...], keep_groups: tuple[str, ...]) -> str:
    parts = []
    if keep_groups:
        parts.append("only_" + "+".join(keep_groups))
    if drop_groups:
        parts.append("no_" + "+".join(drop_groups))
    return "_".join(parts)


def run_baseline(
    root: str | Path,
    source: str,
    model: str,
    seed: int,
    *,
    train_cap: int = 150_000,
    val_cap: int = 50_000,
    fpr_target: float = 0.01,
    row_stride: int = 1,
    targets: list[str] | None = None,
    min_group_flows: int = 500,
    drop_groups: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train and evaluate one baseline; returns a JSON-friendly results dict."""
    root = Path(root)
    seed_everything(seed)
    keep = kept_feature_indices(drop_groups, keep_groups)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    pairs = [
        p
        for p in protocol["pairs"]
        if p["source"] == source and (targets is None or p["target"] in targets)
    ]
    if not pairs:
        raise ValueError(f"No protocol pairs for source {source!r}")

    plan = load_plan(root / "splits" / f"{source}.json")
    scaler = load_scaler(root / "scalers" / f"{source}.json")

    log(f"[{model}/{source}/s{seed}] sampling train (cap {train_cap}/category)")
    train = sample_split(root / source, plan, "train", scaler, train_cap, seed)
    val = sample_split(root / source, plan, "val", scaler, val_cap, seed + 1)
    log(f"  train rows={len(train['y']):,} val rows={len(val['y']):,}")

    est = make_baseline(model, seed)
    est.fit(train["X"][:, keep], train["y"])

    val_score = est.predict_proba(val["X"][:, keep])[:, 1]
    threshold = threshold_at_fpr(val["y"], val_score, fpr_target)
    val_report = evaluate_arrays(
        val["y"], (val_score > threshold).astype(np.uint8), val["group"], y_score=val_score
    )
    log(f"  threshold={threshold:.6f} (source-val FPR target {fpr_target})")

    def predict(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score = est.predict_proba(x[:, keep])[:, 1]
        return (score > threshold).astype(np.uint8), score

    results = []
    for pair in pairs:
        log(f"  evaluating {source} -> {pair['target']} ({pair['eval_set']})")
        results.append(
            evaluate_pair(
                predict, root, pair, min_group_flows=min_group_flows, row_stride=row_stride
            )
        )
    return {
        "meta": {
            "kind": "baseline",
            "model": model,
            "variant": ablation_label(drop_groups, keep_groups),
            "drop_groups": list(drop_groups),
            "keep_groups": list(keep_groups),
            "n_features": len(keep),
            "source": source,
            "seed": seed,
            "train_cap": train_cap,
            "val_cap": val_cap,
            "fpr_target": fpr_target,
            "row_stride": row_stride,
            "threshold": float(threshold),
            "train_rows": int(len(train["y"])),
        },
        "val": val_report,
        "pairs": results,
    }

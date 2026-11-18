"""Evaluation harness: run any model over a protocol pair and report metrics.

A model is any callable `predict(X) -> (y_pred, y_score)` where X is a float32
array (n, 55) of scaled features, y_pred is 0/1 and y_score is a continuous attack
score (or None). For the RL agent, y_score is Q(attack) - Q(benign). The harness
streams the evaluation set block by block, applies the SOURCE scaler (never target
statistics), measures inference time, and returns a JSON-friendly report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nidsrl.data.loading import iter_split
from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.eval.metrics import evaluate_arrays

Predictor = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]


def evaluate_pair(
    predict: Predictor,
    root: str | Path,
    pair: dict[str, Any],
    *,
    min_group_flows: int = 500,
    batch_rows: int = 262_144,
    row_stride: int = 1,
) -> dict[str, Any]:
    """Evaluate `predict` on one protocol pair (see data/protocol.json).

    row_stride > 1 evaluates every k-th row of each block (deterministic thinning for
    quick development runs; use 1 for reported numbers).
    """
    root = Path(root)
    scaler = load_scaler(root / "scalers" / f"{pair['source']}.json")
    plan = load_plan(root / "splits" / f"{pair['target']}.json")

    ys, preds, scores, groups = [], [], [], []
    have_scores = True
    predict_seconds = 0.0
    for block in iter_split(root / pair["target"], plan, pair["eval_set"], scaler):
        sl = slice(None, None, row_stride)
        X = block["X"][sl]
        for start in range(0, len(X), batch_rows):
            chunk = X[start : start + batch_rows]
            t0 = time.perf_counter()
            y_pred, y_score = predict(chunk)
            predict_seconds += time.perf_counter() - t0
            preds.append(np.asarray(y_pred, dtype=np.uint8))
            if y_score is None:
                have_scores = False
            else:
                scores.append(np.asarray(y_score, dtype=np.float32))
        ys.append(block["y"][sl])
        groups.append(block["group"][sl])

    y_true = np.concatenate(ys)
    n = len(y_true)
    metrics = evaluate_arrays(
        y_true,
        np.concatenate(preds),
        np.concatenate(groups),
        y_score=np.concatenate(scores) if have_scores and scores else None,
        shared=pair["shared_groups"],
        unseen=pair["unseen_groups"],
        min_group_flows=min_group_flows,
    )
    return {
        "source": pair["source"],
        "target": pair["target"],
        "kind": pair["kind"],
        "eval_set": pair["eval_set"],
        "predict_seconds": predict_seconds,
        "flows_per_second": (n / predict_seconds) if predict_seconds > 0 else None,
        **metrics,
    }


def save_results(results: list[dict[str, Any]] | dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")

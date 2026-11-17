"""Training-only feature scaling.

Only the continuous features (log1p and fixed-scale columns) are standardized;
binary flag and protocol columns pass through untouched. The scaler is fit on the
TRAIN rows of a source dataset only and is then applied unchanged to every other
split and dataset. That is the zero-shot protocol: no target statistics are used.
Scaled values are clipped to +-clip to keep out-of-range target values bounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidsrl.data.prep import FEATURE_COLUMNS, LOG1P_COLUMNS, SCALED_COLUMNS
from nidsrl.data.splits import TRAIN, assign_rows

N_CONTINUOUS = len(LOG1P_COLUMNS) + len(SCALED_COLUMNS)
CONTINUOUS_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS[:N_CONTINUOUS]
DEFAULT_CLIP = 10.0
STD_FLOOR = 1e-6


def fit_scaler(dataset_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Fit mean/std on the plan's TRAIN rows only, streaming block by block (Chan's update)."""
    if plan["role"] != "source":
        raise ValueError("Only source datasets have training rows to fit a scaler on")
    n = 0
    mean = np.zeros(N_CONTINUOUS)
    m2 = np.zeros(N_CONTINUOUS)
    cols = ["start_ms", *CONTINUOUS_COLUMNS]
    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        df = pd.read_parquet(path, columns=cols)
        keep = assign_rows(df["start_ms"].to_numpy(), plan) == TRAIN
        if not keep.any():
            continue
        x = df.loc[keep, list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64)
        nb = x.shape[0]
        mb = x.mean(axis=0)
        m2b = ((x - mb) ** 2).sum(axis=0)
        delta = mb - mean
        total = n + nb
        mean = mean + delta * nb / total
        m2 = m2 + m2b + delta**2 * n * nb / total
        n = total
    if n == 0:
        raise ValueError("Plan has no training rows")
    std = np.sqrt(m2 / n)
    std = np.where(std < STD_FLOOR, 1.0, std)
    return {
        "columns": list(CONTINUOUS_COLUMNS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_flows": int(n),
        "clip": DEFAULT_CLIP,
    }


def apply_scaler(features: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    """Return a copy with continuous columns standardized and clipped."""
    out = features.copy()
    cols = scaler["columns"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    scaled = (out[cols].to_numpy(dtype=np.float64) - mean) / std
    clip = scaler.get("clip", DEFAULT_CLIP)
    out[cols] = np.clip(scaled, -clip, clip).astype(np.float32)
    return out


def save_scaler(scaler: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scaler, indent=2), encoding="utf-8")


def load_scaler(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

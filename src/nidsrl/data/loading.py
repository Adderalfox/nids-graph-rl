"""Streaming access to prepared data by split.

Yields one dict of numpy arrays per hourly block, restricted to the rows of the
requested split according to the split plan. Nothing is ever loaded wholesale, so a
27M-row dataset can be evaluated on a 16 GB machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidsrl.data.prep import FEATURE_COLUMNS
from nidsrl.data.scaling import apply_scaler
from nidsrl.data.splits import SPLIT_NAMES, assign_rows
from nidsrl.data.taxonomy import GROUPS, OTHER

GROUP_INDEX = {g: i for i, g in enumerate(GROUPS)}
SPLITS = (*SPLIT_NAMES, "all")


def iter_split(
    dataset_dir: str | Path,
    plan: dict[str, Any],
    split: str,
    scaler: dict[str, Any] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield per-block arrays for a split: 'train' | 'val' | 'test' | 'all'.

    'all' includes every row (purged rows too) and is used only to evaluate a
    source model on a different, never-trained-on target dataset.
    Keys: X (float32, scaled if a scaler is given), y (uint8), group (int8 index into
    taxonomy.GROUPS), start_ms, src_id, dst_id.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    code = None if split == "all" else SPLIT_NAMES.index(split)
    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        df = pd.read_parquet(path)
        if code is not None:
            df = df.loc[assign_rows(df["start_ms"].to_numpy(), plan) == code]
        if df.empty:
            continue
        feats = df[list(FEATURE_COLUMNS)]
        if scaler is not None:
            feats = apply_scaler(feats, scaler)
        yield {
            "X": feats.to_numpy(dtype=np.float32),
            "y": df["label"].to_numpy(dtype=np.uint8),
            "group": df["attack_group"]
            .map(lambda g: GROUP_INDEX.get(g, GROUP_INDEX[OTHER]))
            .to_numpy(dtype=np.int8),
            "start_ms": df["start_ms"].to_numpy(dtype=np.int64),
            "src_id": df["src_id"].to_numpy(dtype=np.int32),
            "dst_id": df["dst_id"].to_numpy(dtype=np.int32),
        }

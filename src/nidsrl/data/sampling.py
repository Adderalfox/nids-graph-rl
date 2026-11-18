"""Category-capped sampling of a split for fitting classical models.

Training sets are far too large to load whole (ToN train is 17.8M flows). Each
category (benign and every attack group) is capped at `per_category_cap` rows by
independent Bernoulli sampling per row, so small attack groups are kept in full and
huge ones are thinned. Sampling is deterministic given the seed and never touches
rows outside the requested split.

Note: this changes class priors in the sample. Decision thresholds are therefore
chosen on a validation sample (benign rows are sampled uniformly, so the false
positive rate stays unbiased), never taken from the model's default 0.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nidsrl.data.loading import GROUP_INDEX, iter_split
from nidsrl.data.taxonomy import GROUPS


def inclusion_probabilities(plan: dict[str, Any], split: str, per_category_cap: int) -> np.ndarray:
    """Per-group-code inclusion probability: min(1, cap / category_count)."""
    stats = plan["stats"][split]
    counts = np.zeros(len(GROUPS))
    counts[GROUP_INDEX["benign"]] = stats["flows"] - stats["attacks"]
    for g, c in stats["groups"].items():
        counts[GROUP_INDEX[g]] = c
    return np.where(counts > 0, np.minimum(1.0, per_category_cap / np.maximum(counts, 1.0)), 0.0)


def sample_split(
    dataset_dir: str | Path,
    plan: dict[str, Any],
    split: str,
    scaler: dict[str, Any] | None,
    per_category_cap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return {'X', 'y', 'group'} sampled from one split (train/val/test)."""
    if split not in ("train", "val", "test"):
        raise ValueError("sample_split supports train, val and test only")
    if plan["stats"][split]["flows"] == 0:
        raise ValueError(f"Split {split!r} has no rows in this plan")
    p = inclusion_probabilities(plan, split, per_category_cap)
    xs, ys, gs = [], [], []
    for i, block in enumerate(iter_split(dataset_dir, plan, split, scaler)):
        rng = np.random.default_rng([seed, i])
        keep = rng.random(len(block["y"])) < p[block["group"]]
        if keep.any():
            xs.append(block["X"][keep])
            ys.append(block["y"][keep])
            gs.append(block["group"][keep])
    return {"X": np.concatenate(xs), "y": np.concatenate(ys), "group": np.concatenate(gs)}

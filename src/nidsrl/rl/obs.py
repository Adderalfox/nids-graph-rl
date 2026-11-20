"""Observation feature policy: which observation columns an agent may see.

Observation layout (env.py): [mean of 55 features | max of 35 continuous | log count |
6 history numbers]. Dropping a feature group removes it from both the mean and the max block.
"""

from __future__ import annotations

import numpy as np

from nidsrl.data.scaling import N_CONTINUOUS
from nidsrl.models.features import kept_feature_indices
from nidsrl.rl.env import N_CONT, OBS_DIM
from nidsrl.rl.rundata import N_FEATURES

assert N_CONT == N_CONTINUOUS


def obs_keep_indices(
    drop_groups: tuple[str, ...] | list[str] = (),
    keep_groups: tuple[str, ...] | list[str] = (),
) -> np.ndarray:
    """`flowcount` is a pseudo-group: the log flow-count column (a volume confound diagnostic)."""
    drop_count = "flowcount" in drop_groups
    drop_groups = tuple(g for g in drop_groups if g != "flowcount")
    keep_groups = tuple(g for g in keep_groups if g != "flowcount")
    kept = kept_feature_indices(drop_groups, keep_groups)
    mean_idx = list(kept)
    max_idx = [N_FEATURES + i for i in kept if i < N_CONT]
    tail = list(range(N_FEATURES + N_CONT, OBS_DIM))
    if drop_count:
        tail = tail[1:]  # first tail column is log(flow count)
    return np.asarray(mean_idx + max_idx + tail, dtype=np.int64)

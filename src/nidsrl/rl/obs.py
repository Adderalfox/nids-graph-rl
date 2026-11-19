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
    kept = kept_feature_indices(drop_groups, keep_groups)
    mean_idx = list(kept)
    max_idx = [N_FEATURES + i for i in kept if i < N_CONT]
    tail = list(range(N_FEATURES + N_CONT, OBS_DIM))
    return np.asarray(mean_idx + max_idx + tail, dtype=np.int64)

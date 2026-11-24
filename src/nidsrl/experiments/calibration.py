"""Source-validation operating points for Q-agents (same protocol as the supervised baselines).

Raw argmax policies sit wherever the reward puts them (a balanced-error point, FPR of tens of
percent). For comparison with the baselines, the alert threshold on attack_score(Q) is chosen on
SOURCE validation at a flow-weighted FPR budget, scoring pair-windows under a do-nothing policy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nidsrl.experiments.gat_supervised import SnapRun
from nidsrl.experiments.pair_baseline import collect_groups, weighted_threshold_at_fpr
from nidsrl.graph.snapshot import Snapshot
from nidsrl.rl.env import EnvConfig
from nidsrl.rl.rundata import RunData
from nidsrl.rl.targets import attack_score


def q_threshold_flat(
    q_fn: Callable[[np.ndarray], np.ndarray],
    val_runs: list[RunData],
    keep: np.ndarray,
    env_cfg: EnvConfig,
    fpr: float,
    seed: int = 0,
) -> float:
    x, _a, b, w = collect_groups(val_runs, keep, 1.0, seed, env_cfg)
    return weighted_threshold_at_fpr(attack_score(np.asarray(q_fn(x))), b * w, fpr)


def q_threshold_graph(
    q_fn: Callable[[Snapshot], np.ndarray], val_runs: list[SnapRun], fpr: float
) -> float:
    scores, ben = [], []
    for sr in val_runs:
        for snap, b in zip(sr.snaps, sr.B, strict=True):
            scores.append(attack_score(np.asarray(q_fn(snap))))
            ben.append(b)
    return weighted_threshold_at_fpr(np.concatenate(scores), np.concatenate(ben), fpr)

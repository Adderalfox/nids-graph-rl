"""Policies built on a Q-function or a score function (numpy only)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nidsrl.rl.env import BLOCK, HostPairEnv
from nidsrl.rl.targets import attack_score, select_actions


class QPolicy:
    """Epsilon-greedy policy over q_fn(obs) -> (n, 3). eps=0 gives the evaluation policy.

    `keep` selects observation columns (feature policy); the SAME mask must be used for
    training and evaluation.
    """

    def __init__(
        self,
        q_fn: Callable[[np.ndarray], np.ndarray],
        keep: np.ndarray | None = None,
        eps: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.q_fn, self.keep, self.eps = q_fn, keep, eps
        self.rng = rng or np.random.default_rng(0)

    def act(self, obs: dict[str, Any], env: HostPairEnv) -> tuple[np.ndarray, np.ndarray]:
        x = obs["obs"]
        n = len(x)
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        if self.keep is not None:
            x = x[:, self.keep]
        q = np.asarray(self.q_fn(x))
        return select_actions(q, self.eps, self.rng), attack_score(q)


class ThresholdPolicy:
    """score_fn(obs) -> attack score; alert (action `alert_action`) when score > threshold."""

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        threshold: float,
        alert_action: int = BLOCK,
        keep: np.ndarray | None = None,
    ) -> None:
        self.score_fn, self.threshold = score_fn, threshold
        self.alert_action, self.keep = alert_action, keep

    def act(self, obs: dict[str, Any], env: HostPairEnv) -> tuple[np.ndarray, np.ndarray]:
        x = obs["obs"]
        if len(x) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        if self.keep is not None:
            x = x[:, self.keep]
        s = np.asarray(self.score_fn(x), dtype=np.float32)
        return np.where(s > self.threshold, self.alert_action, 0).astype(np.int64), s

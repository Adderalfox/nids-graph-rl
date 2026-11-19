"""Pure-numpy RL maths shared by every agent (unit-tested without torch).

TD target with semi-MDP discounting (the time between a pair's decisions varies):
    y = r + gamma ** dt * (1 - done) * V(s2)
    V(s2) = max_a Q_target(s2, a)                         (DQN)
          = Q_target(s2, argmax_a Q_online(s2, a))        (Double DQN)
"""

from __future__ import annotations

import numpy as np


def td_targets(
    r: np.ndarray,
    done: np.ndarray,
    dt: np.ndarray,
    gamma: float,
    q_next_target: np.ndarray,
    q_next_online: np.ndarray | None = None,
) -> np.ndarray:
    if q_next_online is None:
        v = q_next_target.max(axis=1)
    else:
        a = q_next_online.argmax(axis=1)
        v = q_next_target[np.arange(len(a)), a]
    disc = np.power(float(gamma), np.maximum(dt, 1).astype(np.float64)) * (~done.astype(bool))
    return (r + disc * v).astype(np.float32)


def epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Linear decay from start to end over decay_steps, then constant."""
    if decay_steps <= 0:
        return end
    frac = min(max(step / decay_steps, 0.0), 1.0)
    return start + frac * (end - start)


def select_actions(q: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Epsilon-greedy over rows of q (n, n_actions)."""
    greedy = q.argmax(axis=1).astype(np.int64)
    if eps <= 0 or len(q) == 0:
        return greedy
    explore = rng.random(len(q)) < eps
    random_a = rng.integers(0, q.shape[1], size=len(q))
    return np.where(explore, random_a, greedy).astype(np.int64)


def attack_score(q: np.ndarray) -> np.ndarray:
    """Ranking score for ROC/PR: best alerting action minus monitor."""
    return (q[:, 1:].max(axis=1) - q[:, 0]).astype(np.float32)

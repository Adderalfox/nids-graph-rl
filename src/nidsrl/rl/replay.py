"""Uniform replay buffer (numpy ring buffer, O(1) add and sample).

Stores per-pair transitions produced by `HostPairEnv`: (s, a, r, s2, done, dt), where dt is
the number of windows until the pair's next decision (used for gamma ** dt discounting).
`sample` also returns the buffer indices so a prioritised buffer (Phase 4) can share the API.
"""

from __future__ import annotations

import numpy as np

FIELDS = ("s", "a", "r", "s2", "done", "dt")


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.obs_dim = capacity, obs_dim
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.dt = np.ones(capacity, dtype=np.int64)
        self.size = 0
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, batch: dict[str, np.ndarray]) -> None:
        n = len(batch["a"])
        if n == 0:
            return
        if n > self.capacity:  # keep only the newest `capacity` transitions
            batch = {k: np.asarray(batch[k])[-self.capacity :] for k in FIELDS}
            n = self.capacity
        idx = (self._next + np.arange(n)) % self.capacity
        for k in FIELDS:
            getattr(self, k)[idx] = batch[k]
        self._next = int((self._next + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._rng.integers(0, self.size, size=n)
        out = {k: getattr(self, k)[idx] for k in FIELDS}
        out["idx"] = idx
        return out

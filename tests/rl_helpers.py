"""Tiny numpy linear-Q learner implementing the trainer's Learner protocol (tests only)."""

import numpy as np


class LinearLearner:
    def __init__(self, obs_dim, lr=0.02, tau=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (obs_dim, 3))
        self.b = np.zeros(3)
        self.Wt, self.bt = self.W.copy(), self.b.copy()
        self.lr, self.tau = lr, tau
        self.fits = 0

    def q_values(self, x):
        return x @ self.W + self.b

    def next_q(self, x):
        return x @ self.W + self.b, x @ self.Wt + self.bt

    def fit(self, s, a, y, w=None):
        q = s @ self.W + self.b
        n = len(a)
        err = q[np.arange(n), a] - y
        g = np.clip(err, -1, 1)  # Huber gradient
        gW = np.zeros_like(self.W)
        gb = np.zeros(3)
        for k in range(3):
            m = a == k
            if m.any():
                gW[:, k] = (s[m] * g[m, None]).sum(0) / n
                gb[k] = g[m].sum() / n
        self.W -= self.lr * gW
        self.b -= self.lr * gb
        self.Wt = (1 - self.tau) * self.Wt + self.tau * self.W
        self.bt = (1 - self.tau) * self.bt + self.tau * self.b
        self.fits += 1
        return {"loss": float((g**2).mean()), "td_error": np.abs(err)}

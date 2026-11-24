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


class LinearScorer:
    """Numpy logistic regression on edge features implementing the graph Scorer protocol."""

    def __init__(self, d, lr=0.5):
        self.w, self.b, self.lr = np.zeros(d), 0.0, lr

    def _p(self, x):
        return 1 / (1 + np.exp(-np.clip(x @ self.w + self.b, -30, 30)))

    def fit_batch(self, b, y, w):
        p = self._p(b.x)
        g = (p - y) * w / max(w.sum(), 1e-9)
        self.w -= self.lr * (b.x.T @ g)
        self.b -= self.lr * g.sum()
        return self.eval_loss(b, y, w)[0]

    def eval_loss(self, b, y, w):
        p = np.clip(self._p(b.x), 1e-7, 1 - 1e-7)
        per = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float((per * w).sum() / max(w.sum(), 1e-9)), float(w.sum())

    def score(self, snap):
        return self._p(snap.x).astype(np.float32)

    def state(self):
        return (self.w.copy(), self.b)

    def load(self, st):
        self.w, self.b = st[0].copy(), st[1]


class LinearGraphLearner:
    """Numpy linear Q on edge features implementing the graph-learner protocol (tests only)."""

    def __init__(self, d, lr=0.05, tau=0.05):
        self.W, self.b = np.zeros((d, 3)), np.zeros(3)
        self.Wt, self.bt = self.W.copy(), self.b.copy()
        self.lr, self.tau, self.fits = lr, tau, 0

    def q_snapshot(self, snap):
        return snap.x @ self.W + self.b

    def next_q(self, g):
        return g.x @ self.W + self.b, g.x @ self.Wt + self.bt

    def fit(self, g, rows, a, y, w=None):
        s = g.x[rows]
        err = (s @ self.W + self.b)[np.arange(len(a)), a] - y
        gr = np.clip(err, -1, 1)
        for k in range(3):
            m = a == k
            if m.any():
                self.W[:, k] -= self.lr * (s[m] * gr[m, None]).sum(0) / len(a)
                self.b[k] -= self.lr * gr[m].sum() / len(a)
        self.Wt = (1 - self.tau) * self.Wt + self.tau * self.W
        self.bt = (1 - self.tau) * self.bt + self.tau * self.b
        self.fits += 1
        return {"loss": float((gr**2).mean()), "td_error": np.abs(err)}

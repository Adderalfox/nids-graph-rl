"""Backend-independent DQN training loop.

The loop, replay, epsilon schedule and TD targets are plain numpy and are unit-tested with a
tiny linear learner. A torch backend (`dqn.py`) only has to implement `Learner`:

    q_values(x)      online-network Q(s, .) for rows x, shape (n, 3), no gradient, eval mode
    next_q(x)        (online Q, target Q) for the next states
    fit(s, a, y, w)  ONE gradient step towards targets y on the actions taken; the target
                     network update (Polyak) happens inside. Returns {"loss", "td_error"}.

Training and validation use the SAME feature mask (`keep`). Model selection uses the source
validation split only (ADR 0004/0005).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from nidsrl.rl.env import HostPairEnv
from nidsrl.rl.policy import QPolicy
from nidsrl.rl.replay import ReplayBuffer
from nidsrl.rl.rundata import RunData
from nidsrl.rl.targets import epsilon, td_targets


class Learner(Protocol):
    def q_values(self, x: np.ndarray) -> np.ndarray: ...
    def next_q(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def fit(
        self, s: np.ndarray, a: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
    ) -> dict[str, Any]: ...


@dataclass
class TrainConfig:
    epochs: int = 8
    max_updates: int = 0  # 0 = no cap
    batch_size: int = 256
    updates_per_window: int = 2
    learn_start: int = 5_000
    buffer_capacity: int = 300_000
    gamma: float = 0.9  # 0 = contextual bandit control
    double: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_windows: int = 30_000
    val_every_epochs: int = 1
    patience: int = 3  # validation rounds without improvement before stopping (0 = off)
    val_metric: str = "macro_f1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunCache:
    """Loads runs on demand and keeps them in RAM up to an approximate byte budget."""

    def __init__(self, loader: Callable[[int], RunData], max_bytes: int = 3 * 2**30) -> None:
        self.loader, self.max_bytes = loader, max_bytes
        self._store: OrderedDict[int, tuple[RunData, int]] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _size(run: RunData) -> int:
        return int(run.g_mean.nbytes + run.g_max.nbytes + run.g_cats.nbytes + 10 * run.g_win.nbytes)

    def get(self, run_id: int) -> RunData:
        if run_id in self._store:
            self._store.move_to_end(run_id)
            return self._store[run_id][0]
        run = self.loader(run_id)
        size = self._size(run)
        self._store[run_id] = (run, size)
        self._bytes += size
        while self._bytes > self.max_bytes and len(self._store) > 1:
            _, (_, s) = self._store.popitem(last=False)
            self._bytes -= s
        return run


def _masked(tr: dict[str, np.ndarray], keep: np.ndarray | None) -> dict[str, np.ndarray]:
    if keep is None:
        return tr
    return {**tr, "s": tr["s"][:, keep], "s2": tr["s2"][:, keep]}


def train_dqn(
    learner: Learner,
    env: HostPairEnv,
    run_ids: list[int],
    get_run: Callable[[int], RunData],
    cfg: TrainConfig,
    *,
    obs_dim: int,
    keep: np.ndarray | None = None,
    seed: int = 0,
    validate: Callable[[QPolicy], dict[str, Any]] | None = None,
    on_best: Callable[[dict[str, Any]], None] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Train over `epochs` passes of shuffled runs. Returns one history record per epoch."""
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(cfg.buffer_capacity, obs_dim, seed)
    train_policy = QPolicy(learner.q_values, keep, eps=1.0, rng=rng)
    windows = updates = 0
    best, bad_rounds = -np.inf, 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        order = list(rng.permutation(run_ids))
        losses: list[float] = []
        action_counts = np.zeros(3, dtype=np.int64)
        stop = False
        for rid in order:
            run = get_run(int(rid))
            if run.n_groups == 0:
                continue
            env.reset(run)
            while not env.done:
                obs = env.observe()
                if obs["transitions"] is not None:
                    buf.add(_masked(obs["transitions"], keep))
                train_policy.eps = epsilon(
                    windows, cfg.eps_start, cfg.eps_end, cfg.eps_decay_windows
                )
                actions, scores = train_policy.act(obs, env)
                env.step(actions, scores)
                action_counts += np.bincount(actions, minlength=3)
                windows += 1
                if len(actions) and len(buf) >= max(cfg.learn_start, cfg.batch_size):
                    for _ in range(cfg.updates_per_window):
                        b = buf.sample(cfg.batch_size)
                        q_on, q_tg = learner.next_q(b["s2"])
                        y = td_targets(
                            b["r"], b["done"], b["dt"], cfg.gamma, q_tg,
                            q_on if cfg.double else None,
                        )  # fmt: skip
                        info = learner.fit(b["s"], b["a"], y, None)
                        losses.append(float(info["loss"]))
                        updates += 1
                if cfg.max_updates and updates >= cfg.max_updates:
                    stop = True
                    break
            final = env.finish().final_transitions
            if len(final["a"]):
                buf.add(_masked(final, keep))
            if stop:
                break

        rec: dict[str, Any] = {
            "epoch": epoch,
            "windows": windows,
            "updates": updates,
            "eps": train_policy.eps,
            "loss": float(np.mean(losses)) if losses else None,
            "train_action_share": (action_counts / max(action_counts.sum(), 1)).tolist(),
            "buffer": len(buf),
        }
        if validate is not None and epoch % cfg.val_every_epochs == 0:
            rep = validate(QPolicy(learner.q_values, keep, eps=0.0))
            score = rep["binary"][cfg.val_metric]
            score = -np.inf if score is None else float(score)
            rec["val"] = {
                "metric": score,
                "tpr": rep["binary"]["tpr"],
                "fpr": rep["binary"]["fpr"],
                "roc_auc": rep["ranking"]["roc_auc"],
                "prevented": rep["response"]["prevented_attack_share"],
                "collateral": rep["response"]["collateral_benign_share"],
            }
            if score > best:
                best, bad_rounds = score, 0
                rec["best"] = True
                if on_best is not None:
                    on_best(rec)
            else:
                bad_rounds += 1
        history.append(rec)
        log(
            f"  epoch {epoch}: updates={updates:,} eps={rec['eps']:.2f} loss={rec['loss']} "
            f"actions(mon/insp/block)={np.round(rec['train_action_share'], 2).tolist()} "
            f"val={rec.get('val')}"
        )
        if stop or (cfg.patience and bad_rounds >= cfg.patience):
            break
    return history

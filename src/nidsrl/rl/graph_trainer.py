"""DQN training loop for graph agents (window-snapshot replay). Backend-independent.

Learner protocol (numpy in/out):
    q_snapshot(snap)            -> (E, 3) online Q for one window graph (eval mode, no grad)
    next_q(graph)               -> (online Q, target Q), each (E_total, 3), no grad
    fit(graph, rows, a, y, w)   -> one gradient step on Q(graph)[rows, a] towards y
Targets use the same numpy `td_targets` (semi-MDP discount, optional Double DQN) as the MLP agent.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from nidsrl.graph.snapshot import GraphBatch, Snapshot, build_snapshot
from nidsrl.rl.env import HostPairEnv
from nidsrl.rl.graph_replay import GraphReplay, GraphRunBuilder, UpdateBatch
from nidsrl.rl.policy import GraphQPolicy
from nidsrl.rl.rundata import RunData
from nidsrl.rl.targets import attack_score, epsilon, select_actions, td_targets
from nidsrl.rl.trainer import TrainConfig


class GraphLearner(Protocol):
    def q_snapshot(self, snap: Snapshot) -> np.ndarray: ...
    def next_q(self, graph: GraphBatch) -> tuple[np.ndarray, np.ndarray]: ...
    def fit(
        self, graph: GraphBatch, rows: np.ndarray, a: np.ndarray, y: np.ndarray,
        w: np.ndarray | None = None,
    ) -> dict[str, Any]: ...  # fmt: skip


def batch_targets(
    learner: GraphLearner, b: UpdateBatch, gamma: float, double: bool
) -> np.ndarray:
    n = len(b.a)
    q_on = np.zeros((n, 3), dtype=np.float32)
    q_tg = np.zeros((n, 3), dtype=np.float32)
    if b.next_graph is not None:
        on, tg = learner.next_q(b.next_graph)
        live = ~b.done
        q_on[live], q_tg[live] = on[b.next_rows[live]], tg[b.next_rows[live]]
    return td_targets(b.r, b.done, b.dt, gamma, q_tg, q_on if double else None)


def train_graph_dqn(
    learner: GraphLearner,
    env: HostPairEnv,
    run_ids: list[int],
    get_run: Callable[[int], RunData],
    cfg: TrainConfig,
    *,
    keep: np.ndarray | None = None,
    windows_per_update: int = 8,
    seed: int = 0,
    validate: Callable[[GraphQPolicy], dict[str, Any]] | None = None,
    on_best: Callable[[dict[str, Any]], None] | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    replay = GraphReplay(cfg.buffer_capacity, seed)
    policy = GraphQPolicy(learner.q_snapshot, keep, eps=1.0, rng=rng)
    windows = updates = 0
    best, bad = -np.inf, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        losses: list[float] = []
        counts = np.zeros(3, dtype=np.int64)
        stop = False
        for rid in rng.permutation(run_ids):
            run = get_run(int(rid))
            if run.n_groups == 0:
                continue
            builder = GraphRunBuilder()
            env.reset(run)
            while not env.done:
                obs = env.observe()
                n = len(obs["gids"])
                if n:
                    snap = build_snapshot(obs["obs"], obs["src"], obs["dst"], keep)
                    builder.add_snapshot(obs["t"], snap)
                    if obs["transitions"] is not None:
                        builder.add_transitions(obs["transitions"])
                    policy.eps = epsilon(windows, cfg.eps_start, cfg.eps_end, cfg.eps_decay_windows)
                    q = learner.q_snapshot(snap)
                    actions = select_actions(q, policy.eps, rng)
                    scores = attack_score(q)
                else:
                    actions, scores = np.zeros(0, np.int64), np.zeros(0, np.float32)
                env.step(actions, scores)
                counts += np.bincount(actions, minlength=3)
                windows += 1
                if n and len(replay) >= cfg.learn_start:
                    for _ in range(cfg.updates_per_window):
                        b = replay.sample(windows_per_update)
                        y = batch_targets(learner, b, cfg.gamma, cfg.double)
                        info = learner.fit(b.graph, b.rows, b.a, y, None)
                        losses.append(float(info["loss"]))
                        updates += 1
                if cfg.max_updates and updates >= cfg.max_updates:
                    stop = True
                    break
            builder.add_transitions(env.finish().final_transitions, final=True)
            replay.add(builder.build())
            if stop:
                break
        rec: dict[str, Any] = {
            "epoch": epoch, "windows": windows, "updates": updates, "eps": policy.eps,
            "loss": float(np.mean(losses)) if losses else None,
            "train_action_share": (counts / max(counts.sum(), 1)).tolist(),
            "replay": len(replay),
        }  # fmt: skip
        if validate is not None and epoch % cfg.val_every_epochs == 0:
            rep = validate(GraphQPolicy(learner.q_snapshot, keep, eps=0.0))
            score = rep["binary"][cfg.val_metric]
            score = -np.inf if score is None else float(score)
            rec["val"] = {
                "metric": score, "tpr": rep["binary"]["tpr"], "fpr": rep["binary"]["fpr"],
                "roc_auc": rep["ranking"]["roc_auc"],
                "prevented": rep["response"]["prevented_attack_share"],
                "collateral": rep["response"]["collateral_benign_share"],
            }  # fmt: skip
            if score > best:
                best, bad = score, 0
                rec["best"] = True
                if on_best is not None:
                    on_best(rec)
            else:
                bad += 1
        history.append(rec)
        shares = np.round(rec["train_action_share"], 2).tolist()
        log(f"  epoch {epoch}: updates={updates:,} eps={rec['eps']:.2f} loss={rec['loss']} "
            f"actions={shares} val={rec.get('val')}")
        if stop or (cfg.patience and bad >= cfg.patience):
            break
    return history

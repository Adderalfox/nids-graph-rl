"""Experiment G: supervised graph attention encoder on window graphs (no RL).

Isolates the value of the graph: same observations, labels, sample weights, threshold rule and
environment as the pair-level gradient-boosting baseline, with message passing over the window
graph. `layers=0` gives the edge-MLP ablation (no neighbourhood information).

Training samples are pair-windows seen under a do-nothing policy (full graphs, no sub-sampling).
Epoch selection uses SOURCE validation loss; the alert threshold uses source validation at a
flow-weighted FPR budget (fallback p=0.5 if saturated). No target data is used before evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from nidsrl.experiments.pair_baseline import weighted_threshold_at_fpr
from nidsrl.experiments.rl_common import SourceData, ablation_tag, source_env_cfg, source_pairs
from nidsrl.graph.snapshot import GraphBatch, Snapshot, build_snapshot, collate
from nidsrl.rl.env import BLOCK, INSPECT, EnvConfig, HostPairEnv
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import GraphThresholdPolicy
from nidsrl.rl.rollout import evaluate_loaded_runs, evaluate_pair_rl
from nidsrl.rl.rundata import RunData
from nidsrl.rl.trainer import RunCache
from nidsrl.utils.seed import seed_everything


@dataclass
class SnapRun:
    snaps: list[Snapshot]
    A: list[np.ndarray]  # attack flows per edge, per window
    B: list[np.ndarray]

    def nbytes(self) -> int:
        edges = sum(s.nbytes() for s in self.snaps)
        labels = sum(a.nbytes + b.nbytes for a, b in zip(self.A, self.B, strict=True))
        return edges + labels


def collect_snap_run(env: HostPairEnv, run: RunData, keep: np.ndarray | None) -> SnapRun:
    """Window graphs of one run under a do-nothing policy (nothing is held)."""
    snaps: list[Snapshot] = []
    A: list[np.ndarray] = []
    B: list[np.ndarray] = []
    env.reset(run)
    while not env.done:
        obs = env.observe()
        n = len(obs["gids"])
        if n:
            snaps.append(build_snapshot(obs["obs"], obs["src"], obs["dst"], keep))
            a, b = env.current_labels()
            A.append(a.astype(np.float64))
            B.append(b.astype(np.float64))
        env.step(np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float32))
    env.finish()
    return SnapRun(snaps, A, B)


def make_batches(
    sr: SnapRun, windows_per_batch: int, ua: float, ub: float, rng: np.random.Generator | None
) -> list[tuple[GraphBatch, np.ndarray, np.ndarray, np.ndarray]]:
    """Batches of whole window graphs -> (graph, label, weight, benign_flows)."""
    idx = np.arange(len(sr.snaps))
    if rng is not None:
        idx = rng.permutation(idx)
    out = []
    for i in range(0, len(idx), windows_per_batch):
        w_idx = idx[i : i + windows_per_batch]
        a = np.concatenate([sr.A[j] for j in w_idx])
        b = np.concatenate([sr.B[j] for j in w_idx])
        y = ((a >= b) & (a > 0)).astype(np.float32)
        w = np.log1p(ua * a + ub * b).astype(np.float32)  # reward-style class balance + log
        out.append((collate([sr.snaps[j] for j in w_idx]), y, w, b))
    return out


class Scorer(Protocol):
    def fit_batch(self, b: GraphBatch, y: np.ndarray, w: np.ndarray) -> float: ...
    def eval_loss(self, b: GraphBatch, y: np.ndarray, w: np.ndarray) -> tuple[float, float]: ...
    def score(self, snap: Snapshot) -> np.ndarray: ...
    def state(self) -> Any: ...
    def load(self, state: Any) -> None: ...


def train_graph_supervised(
    scorer: Scorer,
    train_ids: list[int],
    get_train: Callable[[int], SnapRun],
    val_runs: list[SnapRun],
    ua: float,
    ub: float,
    *,
    epochs: int = 10,
    windows_per_batch: int = 24,
    patience: int = 3,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Epoch loop with validation-loss model selection; restores the best weights at the end."""
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    hist: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        losses = []
        for rid in rng.permutation(train_ids):
            for b, y, w, _ in make_batches(get_train(int(rid)), windows_per_batch, ua, ub, rng):
                losses.append(scorer.fit_batch(b, y, w))
        tot = wsum = 0.0
        for sr in val_runs:
            for b, y, w, _ in make_batches(sr, 4 * windows_per_batch, ua, ub, None):
                loss, ws = scorer.eval_loss(b, y, w)
                tot, wsum = tot + loss * ws, wsum + ws
        val = tot / max(wsum, 1e-9)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val}
        hist.append(rec)
        log(f"  epoch {epoch}: train_loss={rec['train_loss']:.4f} val_loss={val:.4f}")
        if val < best - 1e-5:
            best, best_state, bad = val, scorer.state(), 0
            rec["best"] = True
        else:
            bad += 1
            if patience and bad >= patience:
                break
    if best_state is not None:
        scorer.load(best_state)
    return hist


def val_threshold(
    scorer: Scorer, val_runs: list[SnapRun], ua: float, ub: float, fpr: float
) -> tuple[float, float]:
    """Flow-weighted benign-FPR quantile on source validation (+ saturation fallback)."""
    scores, ben = [], []
    for sr in val_runs:
        for snap, b in zip(sr.snaps, sr.B, strict=True):
            scores.append(scorer.score(snap))
            ben.append(b)
    q = weighted_threshold_at_fpr(np.concatenate(scores), np.concatenate(ben), fpr)
    return (q if q >= 1e-3 else 0.5), q


def run_gat_supervised(
    root: str | Path,
    source: str,
    seed: int,
    *,
    layers: int = 2,
    hidden: int = 64,
    heads: int = 4,
    lr: float = 1e-3,
    epochs: int = 10,
    windows_per_batch: int = 24,
    patience: int = 3,
    drop_groups: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
    fpr_target: float = 0.01,
    inspect_windows: int = 1,
    block_windows: int = 6,
    eval_stride: int = 1,
    targets: list[str] | None = None,
    min_group_flows: int = 500,
    tag: str = "",
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    from nidsrl.graph.snapshot import NODE_DIM
    from nidsrl.models.gat import GraphScorer
    from nidsrl.utils.device import resolve_device

    root = Path(root)
    seed_everything(seed)
    keep = obs_keep_indices(drop_groups, keep_groups)
    data = SourceData(root, source)
    env_cfg: EnvConfig = source_env_cfg(data.plan, inspect_windows, block_windows)
    ua, ub = env_cfg.reward.u_a, env_cfg.reward.u_b
    pairs = source_pairs(root, source, targets)

    def loader(split: str) -> Callable[[int], SnapRun]:
        return lambda rid: collect_snap_run(
            HostPairEnv(env_cfg), data.load_uncached(split, rid), keep
        )

    train_cache = RunCache(loader("train"), 4 * 2**30, lambda sr: sr.nbytes())
    val_cache = RunCache(loader("val"), 2 * 2**30, lambda sr: sr.nbytes())
    train_ids, val_ids = data.ids("train"), data.ids("val")
    log(f"[gat{layers}-sup/{source}/s{seed}] train_runs={len(train_ids)} val_runs={len(val_ids)}")
    val_runs = [val_cache.get(i) for i in val_ids]

    scorer = GraphScorer(len(keep), NODE_DIM, hidden, heads, layers, lr, resolve_device(device))
    hist = train_graph_supervised(
        scorer, train_ids, train_cache.get, val_runs, ua, ub, epochs=epochs,
        windows_per_batch=windows_per_batch, patience=patience, seed=seed, log=log,
    )  # fmt: skip
    thr, q = val_threshold(scorer, val_runs, ua, ub, fpr_target)
    log(f"  threshold={thr:.5f} (val FPR-quantile {q:.6f})")

    name = "edgemlp" if layers == 0 else f"gat{layers}"
    out = {}
    for action_name, action in (("inspect", INSPECT), ("block", BLOCK)):
        policy = GraphThresholdPolicy(scorer.score, thr, action, keep)
        val = evaluate_loaded_runs(
            policy, data.runs("val"), env_cfg=env_cfg, min_group_flows=min_group_flows
        )
        res = []
        for pair in pairs:
            log(f"  [{action_name}] evaluating {source} -> {pair['target']} ({pair['eval_set']})")
            res.append(
                evaluate_pair_rl(
                    policy, root, pair, env_cfg=env_cfg, min_group_flows=min_group_flows,
                    run_stride=eval_stride,
                )  # fmt: skip
            )
        out[action_name] = {
            "meta": {
                "kind": "rl_env",
                "model": f"{name}-sup-{action_name}",
                "variant": ablation_tag(drop_groups, keep_groups, tag),
                "source": source,
                "seed": seed,
                "threshold": thr,
                "threshold_fpr_quantile": q,
                "fpr_target": fpr_target,
                "layers": layers,
                "hidden": hidden,
                "heads": heads,
                "lr": lr,
                "epochs_run": len(hist),
                "n_obs_features": len(keep),
                "eval_stride": eval_stride,
            },
            "history": hist,
            "val": {k: val[k] for k in ("binary", "ranking", "response")},
            "pairs": res,
        }
    return out

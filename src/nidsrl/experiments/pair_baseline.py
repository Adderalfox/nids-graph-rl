"""Supervised baseline INSIDE the host-pair environment (same observations, same actions).

Why: the flow-level baselines (docs/baselines.md) and the pair-level RL agent differ in
representation, label granularity and response semantics at once. This baseline fits
gradient boosting on the SAME 97-number pair-window observations and acts through the SAME
environment, so (RL agent vs this baseline) isolates what reinforcement learning adds.

Training rows: pair-windows seen under a do-nothing policy (no holds, no action history);
label = attack majority (attack flows >= benign flows and > 0); sample weight =
log1p(u_a*A + u_b*B) (the reward's class balancing and log compression), times 1/rate for
sub-sampled non-attack rows. The alert threshold is chosen on SOURCE val at a flow-weighted
false-positive budget (never from target data). Alerting uses inspect or block.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nidsrl.experiments.rl_common import SourceData, ablation_tag, source_env_cfg, source_pairs
from nidsrl.rl.env import BLOCK, INSPECT, OBS_DIM, EnvConfig, HostPairEnv
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import ThresholdPolicy
from nidsrl.rl.rollout import evaluate_loaded_runs, evaluate_pair_rl, rollout_run
from nidsrl.rl.rundata import RunData
from nidsrl.utils.seed import seed_everything


class _Recorder:
    """Do-nothing policy that records pair-window observations and ground truth."""

    def __init__(self, keep: np.ndarray, rate: float, rng: np.random.Generator) -> None:
        self.keep, self.rate, self.rng = keep, rate, rng
        self.x: list[np.ndarray] = []
        self.a: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.w: list[np.ndarray] = []

    def act(self, obs, env):
        n = len(obs["gids"])
        a, b = env.current_labels()
        attack = (a >= b) & (a > 0)
        sel = attack | (self.rng.random(n) < self.rate)
        self.x.append(obs["obs"][sel][:, self.keep])
        self.a.append(a[sel].astype(np.float64))
        self.b.append(b[sel].astype(np.float64))
        self.w.append(np.where(attack[sel], 1.0, 1.0 / self.rate))
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float32)

    def stacked(self):
        return (
            np.concatenate(self.x), np.concatenate(self.a),
            np.concatenate(self.b), np.concatenate(self.w),
        )  # fmt: skip


def collect_groups(runs: list[RunData], keep, rate: float, seed: int, env_cfg: EnvConfig):
    rec = _Recorder(keep, rate, np.random.default_rng(seed))
    env = HostPairEnv(env_cfg)
    for run in runs:
        if run.n_groups:
            rollout_run(env, run, rec)
    return rec.stacked()


def weighted_threshold_at_fpr(
    score: np.ndarray, benign_weight: np.ndarray, fpr: float
) -> float:
    """Score threshold with weighted benign mass above it ~= fpr (alert when score > thr)."""
    m = benign_weight > 0
    s, w = score[m], benign_weight[m]
    if len(s) == 0:
        raise ValueError("Need benign rows to choose a threshold")
    order = np.argsort(s)
    cum = np.cumsum(w[order])
    i = int(np.searchsorted(cum, (1.0 - fpr) * cum[-1], side="left"))
    return float(s[order][min(i, len(s) - 1)])


def run_pair_baseline(
    root: str | Path,
    source: str,
    seed: int,
    *,
    drop_groups: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
    fpr_target: float = 0.01,
    max_iter: int = 200,
    saturated_below: float = 1e-3,
    fallback_threshold: float = 0.5,
    benign_rate: float = 0.2,
    inspect_windows: int = 1,
    block_windows: int = 6,
    eval_stride: int = 1,
    targets: list[str] | None = None,
    min_group_flows: int = 500,
    tag: str = "",
    alert_actions: tuple[str, ...] = ("inspect", "block"),
    group_cache: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Returns {'inspect': results, 'block': results} (one fitted model, two alert actions).

    `group_cache` (a dict owned by the caller) lets many feature policies on the same source
    and seed reuse the collected pair-windows: they are recorded with ALL columns and sliced.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    root = Path(root)
    seed_everything(seed)
    keep = obs_keep_indices(drop_groups, keep_groups)
    data = SourceData(root, source)
    env_cfg = source_env_cfg(data.plan, inspect_windows, block_windows)
    ua, ub = env_cfg.reward.u_a, env_cfg.reward.u_b
    pairs = source_pairs(root, source, targets)

    full = np.arange(OBS_DIM)
    ck = f"{source}/{seed}/{benign_rate}"
    if group_cache is not None and ck in group_cache:
        (x_all, a, b, ws), (xv_all, _av, bv, wv) = group_cache[ck]
    else:
        log(f"[hgb-pair/{source}/s{seed}] collecting train/val pair-windows")
        train_g = collect_groups(data.runs("train"), full, benign_rate, seed, env_cfg)
        val_g = collect_groups(data.runs("val"), full, 1.0, seed + 1, env_cfg)
        (x_all, a, b, ws), (xv_all, _av, bv, wv) = train_g, val_g
        if group_cache is not None:
            group_cache[ck] = (train_g, val_g)
    x, xv = x_all[:, keep], xv_all[:, keep]
    y = ((a >= b) & (a > 0)).astype(np.int64)
    weight = np.log1p(ua * a + ub * b) * ws
    log(f"  train groups={len(y):,} attack-majority={int(y.sum()):,}")
    est = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.1, early_stopping=False, random_state=seed
    )
    est.fit(x, y, sample_weight=weight)

    sv = est.predict_proba(xv)[:, 1]
    thr_fpr = weighted_threshold_at_fpr(sv, bv * wv, fpr_target)
    # A (near-)perfectly separable source puts the FPR quantile on the saturated floor
    # (score ~ 0), so "alert above it" would flag everything on a new dataset. Only in that
    # degenerate case fall back to p=0.5 of the class-balanced model.
    thr = thr_fpr if thr_fpr >= saturated_below else fallback_threshold
    log(f"  threshold={thr:.5f} (val FPR-quantile {thr_fpr:.6f}, fallback {fallback_threshold})")

    def score_fn(obs: np.ndarray) -> np.ndarray:
        return est.predict_proba(obs)[:, 1]

    out = {}
    for name, action in (("inspect", INSPECT), ("block", BLOCK)):
        if name not in alert_actions:
            continue
        policy = ThresholdPolicy(score_fn, thr, action, keep)
        val = evaluate_loaded_runs(
            policy, data.runs("val"), env_cfg=env_cfg, min_group_flows=min_group_flows
        )
        res = []
        for pair in pairs:
            log(f"  [{name}] evaluating {source} -> {pair['target']} ({pair['eval_set']})")
            res.append(
                evaluate_pair_rl(
                    policy, root, pair, env_cfg=env_cfg, min_group_flows=min_group_flows,
                    run_stride=eval_stride,
                )  # fmt: skip
            )
        out[name] = {
            "meta": {
                "kind": "rl_env",
                "model": f"hgb-pair-{name}",
                "variant": ablation_tag(drop_groups, keep_groups, tag),
                "source": source,
                "seed": seed,
                "threshold": thr,
                "threshold_fpr_quantile": thr_fpr,
                "saturated_below": saturated_below,
                "fallback_threshold": fallback_threshold,
                "fpr_target": fpr_target,
                "benign_rate": benign_rate,
                "n_obs_features": len(keep),
                "env": {"inspect_windows": inspect_windows, "block_windows": block_windows},
                "eval_stride": eval_stride,
            },
            "val": {k: val[k] for k in ("binary", "ranking", "response")},
            "pairs": res,
        }
    return out

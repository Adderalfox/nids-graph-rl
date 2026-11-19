"""Rollouts and flow-level evaluation of policies in the host-pair environment.

A policy implements `act(obs, env) -> (actions, scores)`: one action (0 monitor,
1 inspect, 2 block) and one continuous attack score per active pair. Learned policies
use only `obs`; the oracle policies below also read ground truth through `env` and exist
for diagnostics only.

Flow-level accounting (comparable with the supervised baselines): every flow inherits the
alert state of its entity-window. A flow is alerted if its pair's action in that window was
not monitor, or if the pair was held (inspected/blocked) when the flow occurred. Held
attack flows therefore count as detected and held benign flows as false alarms.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.eval.metrics import (
    binary_report_from_counts,
    group_report_weighted,
    ranking_report_weighted,
)
from nidsrl.rl.env import BLOCK, EnvConfig, HostPairEnv, RunResult
from nidsrl.rl.rundata import RunData, load_run, run_ids


class Policy(Protocol):
    def act(self, obs: dict[str, Any], env: HostPairEnv) -> tuple[np.ndarray, np.ndarray]: ...


class NeverPolicy:
    """Always monitor (no response): the do-nothing reference."""

    def act(self, obs: dict[str, Any], env: HostPairEnv) -> tuple[np.ndarray, np.ndarray]:
        n = len(obs["gids"])
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float32)


class OraclePolicy:
    """Diagnostic oracle: acts on ground truth. rule 'any' = any attack flow in the window;
    'majority' = attack flows at least as many as benign. Never use for reported results."""

    def __init__(self, rule: str = "any", action: int = BLOCK) -> None:
        if rule not in ("any", "majority"):
            raise ValueError("rule must be 'any' or 'majority'")
        self.rule, self.action = rule, action

    def act(self, obs: dict[str, Any], env: HostPairEnv) -> tuple[np.ndarray, np.ndarray]:
        a, b = env.current_labels()
        flag = (a > 0) if self.rule == "any" else ((a >= b) & (a > 0))
        score = (a / np.maximum(a + b, 1)).astype(np.float32)
        return np.where(flag, self.action, 0).astype(np.int64), score


def rollout_run(
    env: HostPairEnv,
    run: RunData,
    policy: Policy,
    on_transitions: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> RunResult:
    """Play one run to the end and return its per-group results."""
    env.reset(run)
    while not env.done:
        obs = env.observe()
        if on_transitions is not None and obs["transitions"] is not None:
            on_transitions(obs["transitions"])
        actions, scores = policy.act(obs, env)
        env.step(actions, scores)
    result = env.finish()
    if on_transitions is not None and len(result.final_transitions["a"]):
        on_transitions(result.final_transitions)
    return result


def summarize_results(
    results: list[RunResult],
    shared: list[str] | None = None,
    unseen: list[str] | None = None,
    min_group_flows: int = 500,
) -> dict[str, Any]:
    """Flow-level metrics and response metrics from per-group results of several runs."""
    alert = np.concatenate([r.alert for r in results])
    score = np.concatenate([r.score for r in results])
    held = np.concatenate([r.held for r in results])
    A = np.concatenate([r.A for r in results]).astype(np.float64)
    B = np.concatenate([r.B for r in results]).astype(np.float64)
    cats = np.concatenate([r.cats for r in results])
    total = A + B

    binary = binary_report_from_counts(
        float((A * alert).sum()),
        float((B * alert).sum()),
        float((A * ~alert).sum()),
        float((B * ~alert).sum()),
    )
    response = {
        "prevented_attack_share": float((A * held).sum() / A.sum()) if A.sum() else None,
        "collateral_benign_share": float((B * held).sum() / B.sum()) if B.sum() else None,
        "alerted_flow_share": float((total * alert).sum() / total.sum()) if total.sum() else None,
        "n_decisions": int(sum(r.n_decisions for r in results)),
        "n_inspect": int(sum(r.n_inspect for r in results)),
        "n_block": int(sum(r.n_block for r in results)),
    }
    return {
        "n_flows": int(total.sum()),
        "binary": binary,
        "ranking": ranking_report_weighted(score, A, B),
        "groups": group_report_weighted(alert, cats, shared or [], unseen or [], min_group_flows),
        "response": response,
    }


def evaluate_loaded_runs(
    policy: Policy,
    runs: list[RunData],
    *,
    env_cfg: EnvConfig | None = None,
    shared: list[str] | None = None,
    unseen: list[str] | None = None,
    min_group_flows: int = 500,
) -> dict[str, Any]:
    """Evaluate a policy on already loaded runs (used for validation during training)."""
    env = HostPairEnv(env_cfg)
    results = [rollout_run(env, r, policy) for r in runs if r.n_groups > 0]
    if not results:
        raise ValueError("No runs with data to evaluate")
    return {"runs": len(results), **summarize_results(results, shared, unseen, min_group_flows)}


def evaluate_pair_rl(
    policy: Policy,
    root: str | Path,
    pair: dict[str, Any],
    *,
    env_cfg: EnvConfig | None = None,
    min_group_flows: int = 500,
    run_stride: int = 1,
    window_s: int = 10,
) -> dict[str, Any]:
    """Evaluate a policy on one protocol pair with the SOURCE scaler (zero-shot for transfer).

    run_stride > 1 evaluates every k-th run (quick development passes; use 1 for results).
    """
    root = Path(root)
    scaler = load_scaler(root / "scalers" / f"{pair['source']}.json")
    plan = load_plan(root / "splits" / f"{pair['target']}.json")
    target_dir = root / pair["target"]
    ids = run_ids(target_dir, plan, pair["eval_set"])[::run_stride]

    env = HostPairEnv(env_cfg)
    results = []
    t0 = time.perf_counter()
    for rid in ids:
        run = load_run(target_dir, plan, pair["eval_set"], rid, scaler, window_s)
        if run.n_groups == 0:
            continue
        results.append(rollout_run(env, run, policy))
    seconds = time.perf_counter() - t0
    if not results:
        raise ValueError(f"No data to evaluate for {pair['source']} -> {pair['target']}")
    summary = summarize_results(
        results, pair["shared_groups"], pair["unseen_groups"], min_group_flows
    )
    return {
        "source": pair["source"],
        "target": pair["target"],
        "kind": pair["kind"],
        "eval_set": pair["eval_set"],
        "runs": len(results),
        "seconds": seconds,
        **summary,
    }

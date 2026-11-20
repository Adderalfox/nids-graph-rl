"""Source-only, nested leave-one-dataset-out selection of the feature policy.

For each held-out dataset H, the policy is chosen using ONLY transfer between the other source
datasets (H is neither trained on nor scored). BoT and H never influence the pick for H.
A global policy (mean over all source-source pairs) gives the main configuration; BoT stays a
final untouched check. Proxy model: pair-level gradient boosting (fast, same observations).
The candidate list is a-priori plus the earlier flow-level shortcut diagnostics: disclose that.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nidsrl.experiments.pair_baseline import run_pair_baseline

# name -> groups to drop. host_fingerprint is dropped everywhere except 'all_features'.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "all_features": (),
    "base": ("host_fingerprint",),
    "no_pkt_size": ("host_fingerprint", "pkt_size"),
    "no_tcp_flags": ("host_fingerprint", "tcp_flags"),
    "no_pkt_size_tcp_flags": ("host_fingerprint", "pkt_size", "tcp_flags"),
    "no_proto": ("host_fingerprint", "proto"),
    "no_volume": ("host_fingerprint", "volume"),
    "no_timing": ("host_fingerprint", "timing"),
    "no_retrans": ("host_fingerprint", "retrans"),
}


def nested_picks(
    auc: dict[tuple[str, str, str], float], sources: list[str], candidates: list[str]
) -> dict[str, Any]:
    """auc[(candidate, source, target)] -> ROC-AUC. Returns per-held-out picks and a global pick."""

    def score(cand: str, srcs: list[str]) -> float:
        vals = [
            auc[(cand, s, t)]
            for s, t in itertools.permutations(srcs, 2)
            if (cand, s, t) in auc and auc[(cand, s, t)] is not None
        ]
        return float(np.mean(vals)) if vals else float("-inf")

    def best(srcs: list[str]) -> tuple[str, dict[str, float]]:
        sc = {c: score(c, srcs) for c in candidates}
        return max(candidates, key=lambda c: (sc[c], -candidates.index(c))), sc

    picks = {}
    for h in sources:
        others = [s for s in sources if s != h]
        pick, sc = best(others)
        picks[h] = {"pick": pick, "scores": sc, "selected_on": others}
    gpick, gsc = best(sources)
    return {"per_heldout": picks, "global": {"pick": gpick, "scores": gsc}}


def run_selection(
    root: str | Path,
    sources: list[str],
    seed: int = 0,
    *,
    candidates: dict[str, tuple[str, ...]] | None = None,
    eval_stride: int = 10,
    min_group_flows: int = 500,
    max_iter: int = 200,
    done: dict[str, Any] | None = None,
    on_result: Callable[[str, dict[str, Any]], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Runs every candidate on every source, evaluating on the OTHER SOURCES only.

    `done` maps "cand|source" to a stored result so an interrupted run can resume.
    """
    cands = candidates or CANDIDATES
    results: dict[str, Any] = dict(done or {})
    for source in sources:
        cache: dict[str, Any] = {}
        others = [s for s in sources if s != source]
        for name, drop in cands.items():
            key = f"{name}|{source}"
            if key in results:
                continue
            out = run_pair_baseline(
                root, source, seed, drop_groups=drop, targets=others + [source],
                eval_stride=eval_stride, min_group_flows=min_group_flows,
                alert_actions=("block",), group_cache=cache, max_iter=max_iter, log=lambda *_: None,
            )  # fmt: skip
            pairs = {
                p["target"]: {
                    "roc_auc": p["ranking"]["roc_auc"],
                    "tpr_at_fpr_1pct": p["ranking"].get("tpr_at_fpr_0.01"),
                }
                for p in out["block"]["pairs"]
            }
            results[key] = pairs
            log(f"  {key}: " + ", ".join(f"{t}={v['roc_auc']:.3f}" for t, v in pairs.items()))
            if on_result:
                on_result(key, pairs)
    auc = {}
    for key, pairs in results.items():
        cand, src = key.split("|")
        for tgt, v in pairs.items():
            if tgt != src and v["roc_auc"] is not None:
                auc[(cand, src, tgt)] = v["roc_auc"]
    return {"raw": results, **nested_picks(auc, sources, list(cands))}

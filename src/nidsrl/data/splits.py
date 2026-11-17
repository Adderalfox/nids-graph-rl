"""Phase 1c: stratified, purged train/val/test split planning.

Why not a plain chronological split: in these datasets attack families are scheduled
in time (e.g. CIC runs a different attack each day), so a chronological cut leaves
whole families out of train/val. Instead the timeline is cut into short runs
(default 30 min = 6 segments of 5 min). Runs are assigned to train/val/test by a
deterministic greedy stratified assignment so every split gets its share of every
attack group, and `purge_segments` segments are dropped wherever the split changes,
so no train and test flow can share a window, a long-lived flow or temporal state.

Caveat (documented in ADR 0005): the same attack campaign can appear in train and
test (in different, purged-apart runs), so in-dataset scores are an optimistic
reference. The headline metric is zero-shot transfer on the whole target dataset.

Runs are also the natural episode unit for the temporal model: sequences never
cross a run boundary, and temporal state resets at each run start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENIGN = "benign"
TRAIN, VAL, TEST, PURGED = 0, 1, 2, -1
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEG_MS = 300_000  # 5 minutes; divides the 1-hour block size


@dataclass
class SegStat:
    seg: int
    n: int
    groups: dict[str, int] = field(default_factory=dict)  # attack groups only

    @property
    def n_attack(self) -> int:
        return sum(self.groups.values())


@dataclass
class SegmentStats:
    seg_ms: int
    segs: list[SegStat]


def compute_segment_stats(dataset_dir: str | Path, seg_ms: int = DEFAULT_SEG_MS) -> SegmentStats:
    """Per-segment flow and attack-group counts, in time order."""
    acc: dict[int, SegStat] = {}
    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        df = pd.read_parquet(path, columns=["start_ms", "attack_group"])
        seg = (df["start_ms"].to_numpy() // seg_ms).astype(np.int64)
        totals = pd.Series(seg).value_counts()
        for s, n in totals.items():
            acc.setdefault(int(s), SegStat(int(s), 0)).n += int(n)
        attack = df["attack_group"] != BENIGN
        if attack.any():
            counts = pd.DataFrame(
                {"seg": seg[attack.to_numpy()], "g": df.loc[attack, "attack_group"]}
            )
            for (s, g), n in counts.groupby(["seg", "g"]).size().items():
                st = acc[int(s)]
                st.groups[str(g)] = st.groups.get(str(g), 0) + int(n)
    return SegmentStats(seg_ms, [acc[k] for k in sorted(acc)])


def _summarize(segs: list[SegStat]) -> dict[str, Any]:
    flows = sum(s.n for s in segs)
    attacks = sum(s.n_attack for s in segs)
    groups: dict[str, int] = {}
    for s in segs:
        for g, c in s.groups.items():
            groups[g] = groups.get(g, 0) + c
    return {
        "segments": len(segs),
        "flows": flows,
        "attacks": attacks,
        "attack_rate": (attacks / flows) if flows else 0.0,
        "groups": dict(sorted(groups.items())),
    }


def plan_split(
    stats: SegmentStats,
    *,
    run_segments: int = 6,
    purge_segments: int = 1,
    fractions: tuple[float, float, float] = (0.70, 0.10, 0.20),
    min_group_flows: int = 100,
) -> dict[str, Any]:
    """Assign runs to train/val/test with stratification over benign and attack groups."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    segs = stats.segs
    cats = [BENIGN] + sorted({g for s in segs for g in s.groups})
    cidx = {c: i for i, c in enumerate(cats)}

    run_counts: dict[int, np.ndarray] = {}
    for s in segs:
        vec = run_counts.setdefault(s.seg // run_segments, np.zeros(len(cats)))
        vec[0] += s.n - s.n_attack
        for g, c in s.groups.items():
            vec[cidx[g]] += c
    run_ids = sorted(run_counts)
    if len(run_ids) < 6:
        raise ValueError(f"Only {len(run_ids)} runs; too few to split three ways")

    totals = np.sum([run_counts[r] for r in run_ids], axis=0)
    safe_tot = np.where(totals > 0, totals, 1.0)
    frac = np.asarray(fractions, dtype=float)
    fill = np.zeros((3, len(cats)))

    def priority(r: int) -> float:
        return float(np.max(run_counts[r] / safe_tot))

    assignment: dict[int, int] = {}
    for r in sorted(run_ids, key=lambda r: (-priority(r), r)):
        w = run_counts[r] / safe_tot
        scores = []
        for s in range(3):
            deficit = 1.0 - fill[s] / (frac[s] * safe_tot)
            scores.append(float(np.sum(w * deficit)) - 1e-9 * (0, 2, 1)[s])
        best = int(np.argmax(scores))
        assignment[r] = best
        fill[best] += run_counts[r]

    seg_split = {s.seg: assignment[s.seg // run_segments] for s in segs}
    present = set(seg_split)
    purged = sorted(
        s
        for s in seg_split
        if any(
            (s - k) in present and seg_split[s - k] != seg_split[s]
            for k in range(1, purge_segments + 1)
        )
    )
    purged_set = set(purged)

    buckets: dict[str, list[SegStat]] = {n: [] for n in SPLIT_NAMES}
    for s in segs:
        if s.seg not in purged_set:
            buckets[SPLIT_NAMES[seg_split[s.seg]]].append(s)
    split_stats = {n: _summarize(v) for n, v in buckets.items()}
    split_stats["all"] = _summarize(segs)

    warnings = []
    for g in cats[1:]:
        total_g = split_stats["all"]["groups"].get(g, 0)
        if total_g < 3 * min_group_flows:
            continue
        for n in SPLIT_NAMES:
            if split_stats[n]["groups"].get(g, 0) < min_group_flows:
                warnings.append(f"group '{g}' has fewer than {min_group_flows} flows in {n}")

    return {
        "role": "source",
        "seg_ms": stats.seg_ms,
        "run_segments": run_segments,
        "purge_segments": purge_segments,
        "fractions": list(fractions),
        "runs": {
            n: sorted(r for r, a in assignment.items() if a == i) for i, n in enumerate(SPLIT_NAMES)
        },
        "purged_segments": purged,
        "stats": split_stats,
        "warnings": warnings,
    }


def plan_target_only(stats: SegmentStats) -> dict[str, Any]:
    """Target-only dataset: every row is test; nothing is used for training."""
    summary = _summarize(stats.segs)
    empty = _summarize([])
    return {
        "role": "target_only",
        "seg_ms": stats.seg_ms,
        "run_segments": 0,
        "purge_segments": 0,
        "fractions": [0.0, 0.0, 1.0],
        "runs": {n: [] for n in SPLIT_NAMES},
        "purged_segments": [],
        "stats": {"train": empty, "val": empty, "test": summary, "all": summary},
        "warnings": [],
    }


def assign_rows(start_ms: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    """Return a split code per row: 0 train, 1 val, 2 test, -1 purged/unassigned."""
    start_ms = np.asarray(start_ms, dtype=np.int64)
    if plan["role"] == "target_only":
        return np.full(start_ms.shape, TEST, dtype=np.int8)
    seg = start_ms // plan["seg_ms"]
    run = seg // plan["run_segments"]
    ids, codes = [], []
    for code, name in enumerate(SPLIT_NAMES):
        ids.extend(plan["runs"][name])
        codes.extend([code] * len(plan["runs"][name]))
    order = np.argsort(ids)
    ids_a, codes_a = np.asarray(ids, dtype=np.int64)[order], np.asarray(codes, dtype=np.int8)[order]
    pos = np.clip(np.searchsorted(ids_a, run), 0, len(ids_a) - 1)
    out = np.where(ids_a[pos] == run, codes_a[pos], PURGED).astype(np.int8)
    if plan["purged_segments"]:
        out[np.isin(seg, np.asarray(plan["purged_segments"], dtype=np.int64))] = PURGED
    return out


def save_plan(plan: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plan, indent=2), encoding="utf-8")


def load_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

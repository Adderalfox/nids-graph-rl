"""Pieces shared by the RL-environment experiments (agent and supervised pair baseline)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.rl.env import EnvConfig, RewardConfig, class_weights
from nidsrl.rl.rundata import RunData, load_run, run_ids
from nidsrl.rl.trainer import RunCache


def source_pairs(root: Path, source: str, targets: list[str] | None) -> list[dict[str, Any]]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    pairs = [
        p
        for p in protocol["pairs"]
        if p["source"] == source and (targets is None or p["target"] in targets)
    ]
    if not pairs:
        raise ValueError(f"No protocol pairs for source {source!r}")
    return pairs


def source_env_cfg(plan: dict[str, Any], inspect_windows: int, block_windows: int) -> EnvConfig:
    """Environment whose class weights come from SOURCE TRAIN only."""
    ua, ub = class_weights(plan, "train")
    return EnvConfig(inspect_windows, block_windows, RewardConfig(u_a=ua, u_b=ub))


class SourceData:
    """Cached access to the runs of one source dataset (scaled with its own scaler)."""

    def __init__(self, root: Path, source: str, train_bytes: int = 4 * 2**30) -> None:
        self.root, self.source = root, source
        self.plan = load_plan(root / "splits" / f"{source}.json")
        self.scaler = load_scaler(root / "scalers" / f"{source}.json")
        self._caches = {
            s: RunCache(lambda rid, s=s: self._load(s, rid), b)
            for s, b in (("train", train_bytes), ("val", 2 * 2**30))
        }

    def _load(self, split: str, rid: int) -> RunData:
        return load_run(self.root / self.source, self.plan, split, rid, self.scaler)

    def ids(self, split: str) -> list[int]:
        return run_ids(self.root / self.source, self.plan, split)

    def runs(self, split: str, stride: int = 1) -> list[RunData]:
        return [self._caches[split].get(i) for i in self.ids(split)[::stride]]

    def get(self, split: str):
        return self._caches[split].get


def ablation_tag(drop_groups, keep_groups, tag: str = "") -> str:
    parts = []
    if keep_groups:
        parts.append("only_" + "+".join(keep_groups))
    if drop_groups:
        parts.append("no_" + "+".join(drop_groups))
    if tag:
        parts.append(tag)
    return "_".join(parts)

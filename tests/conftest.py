import json

import pytest
from helpers import CAMPAIGNS, make_prepared

from nidsrl.data.protocol import build_protocol
from nidsrl.data.scaling import fit_scaler, save_scaler
from nidsrl.data.splits import (
    compute_segment_stats,
    plan_split,
    plan_target_only,
    save_plan,
)


@pytest.fixture
def world(tmp_path):
    """Three synthetic prepared datasets (a, b sources; c target-only) with plans,
    scalers and protocol.json written under tmp_path (the 'processed' root)."""
    a = make_prepared(tmp_path, "a", CAMPAIGNS, seed=1)
    b = make_prepared(tmp_path, "b", {"dos": range(0, 12), "web": range(12, 24)}, seed=2)
    c = make_prepared(tmp_path, "c", {"dos": range(0, 6)}, n_blocks=6, seed=3)
    plans = {
        "a": plan_split(compute_segment_stats(a)),
        "b": plan_split(compute_segment_stats(b)),
        "c": plan_target_only(compute_segment_stats(c)),
    }
    for name, plan in plans.items():
        save_plan(plan, tmp_path / "splits" / f"{name}.json")
    for name in ("a", "b"):
        save_scaler(fit_scaler(tmp_path / name, plans[name]), tmp_path / "scalers" / f"{name}.json")
    protocol = build_protocol(plans, sources=("a", "b"), min_group_flows=50)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    pairs = {(p["source"], p["target"]): p for p in protocol["pairs"]}
    return tmp_path, plans, pairs

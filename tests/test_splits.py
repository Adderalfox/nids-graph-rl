import numpy as np
import pandas as pd
import pytest
from helpers import CAMPAIGNS, load_all, make_prepared

from nidsrl.data.prep import FEATURE_COLUMNS
from nidsrl.data.protocol import build_protocol
from nidsrl.data.scaling import CONTINUOUS_COLUMNS, apply_scaler, fit_scaler
from nidsrl.data.splits import (
    PURGED,
    TEST,
    TRAIN,
    VAL,
    assign_rows,
    compute_segment_stats,
    plan_split,
    plan_target_only,
)


def test_every_split_gets_every_group_even_when_campaigns_are_scheduled(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    plan = plan_split(compute_segment_stats(d))
    for split in ("train", "val", "test"):
        assert set(plan["stats"][split]["groups"]) == {"dos", "recon", "web"}, split
    assert plan["warnings"] == []


def test_split_fractions_are_close_to_targets(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    st = plan_split(compute_segment_stats(d))["stats"]
    total = sum(st[s]["flows"] for s in ("train", "val", "test"))
    assert 0.6 <= st["train"]["flows"] / total <= 0.8
    assert 0.05 <= st["val"]["flows"] / total <= 0.16
    assert 0.12 <= st["test"]["flows"] / total <= 0.30


def test_plan_is_deterministic(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    s = compute_segment_stats(d)
    assert plan_split(s) == plan_split(s)


def test_assign_rows_matches_plan_stats_and_purges_boundaries(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    plan = plan_split(compute_segment_stats(d))
    df = load_all(d)
    codes = assign_rows(df["start_ms"].to_numpy(), plan)
    for code, name in ((TRAIN, "train"), (VAL, "val"), (TEST, "test")):
        assert int((codes == code).sum()) == plan["stats"][name]["flows"], name
    assert int((codes == PURGED).sum()) > 0

    # leakage check: segments of different splits are never adjacent
    seg = (df["start_ms"].to_numpy() // plan["seg_ms"]).astype(np.int64)
    seg_code = pd.Series(codes).groupby(seg).first()
    kept = seg_code[seg_code != PURGED]
    for s, c in kept.items():
        neighbour = kept.get(s - 1)
        if neighbour is not None:
            assert neighbour == c, f"segments {s - 1} and {s} are adjacent but in different splits"


def test_too_few_runs_raises(tmp_path):
    d = make_prepared(tmp_path, "a", {"dos": range(0, 2)}, n_blocks=2)
    with pytest.raises(ValueError):
        plan_split(compute_segment_stats(d))


def test_target_only_is_all_test(tmp_path):
    d = make_prepared(tmp_path, "b", {"dos": range(0, 3)}, n_blocks=3)
    plan = plan_target_only(compute_segment_stats(d))
    codes = assign_rows(load_all(d)["start_ms"].to_numpy(), plan)
    assert (codes == TEST).all()
    assert plan["stats"]["train"]["flows"] == 0 and plan["stats"]["test"]["flows"] == 3 * 1200


def test_scaler_is_fit_on_train_rows_only(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    plan = plan_split(compute_segment_stats(d))
    # poison every non-train row with extreme values: a leaky scaler would notice
    for path in sorted((d / "flows").glob("*.parquet")):
        df = pd.read_parquet(path)
        codes = assign_rows(df["start_ms"].to_numpy(), plan)
        df.loc[codes != TRAIN, list(CONTINUOUS_COLUMNS)] = 1e6
        df.to_parquet(path, index=False)

    scaler = fit_scaler(d, plan)
    df = load_all(d)
    codes = assign_rows(df["start_ms"].to_numpy(), plan)
    ref = df.loc[codes == TRAIN, list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64)
    assert scaler["n_flows"] == len(ref)
    assert np.allclose(scaler["mean"], ref.mean(axis=0), atol=1e-6)
    assert np.allclose(scaler["std"], ref.std(axis=0), atol=1e-6)


def test_apply_scaler_standardizes_clips_and_spares_binary_columns(tmp_path):
    d = make_prepared(tmp_path, "a", CAMPAIGNS)
    plan = plan_split(compute_segment_stats(d))
    scaler = fit_scaler(d, plan)
    feats = load_all(d)[list(FEATURE_COLUMNS)].reset_index(drop=True)
    scaled = apply_scaler(feats, scaler)
    cont = scaled[list(CONTINUOUS_COLUMNS)].to_numpy()
    assert np.allclose(cont.mean(axis=0), 0, atol=0.1)
    assert np.allclose(cont.std(axis=0), 1, atol=0.1)
    binary = [c for c in FEATURE_COLUMNS if c not in CONTINUOUS_COLUMNS]
    assert np.allclose(scaled[binary].to_numpy(), feats[binary].to_numpy())
    wild = feats.copy()
    wild[list(CONTINUOUS_COLUMNS)] = 1e9
    assert apply_scaler(wild, scaler)[list(CONTINUOUS_COLUMNS)].to_numpy().max() <= 10.0


def test_scaler_refuses_target_only(tmp_path):
    d = make_prepared(tmp_path, "b", {"dos": range(0, 3)}, n_blocks=3)
    with pytest.raises(ValueError):
        fit_scaler(d, plan_target_only(compute_segment_stats(d)))


def test_protocol_uses_test_for_in_dataset_and_all_for_transfer(tmp_path):
    a = make_prepared(tmp_path, "a", {"dos": range(0, 12), "recon": range(12, 24)}, seed=1)
    b = make_prepared(tmp_path, "b", {"dos": range(0, 12), "web": range(12, 24)}, seed=2)
    c = make_prepared(tmp_path, "c", {"dos": range(0, 6)}, n_blocks=6, seed=3)
    plans = {
        "a": plan_split(compute_segment_stats(a)),
        "b": plan_split(compute_segment_stats(b)),
        "c": plan_target_only(compute_segment_stats(c)),
    }
    proto = build_protocol(plans, sources=("a", "b"), min_group_flows=50)
    pairs = {(p["source"], p["target"]): p for p in proto["pairs"]}
    assert len(pairs) == 6
    assert pairs[("a", "a")]["eval_set"] == "test" and pairs[("a", "a")]["kind"] == "in_dataset"
    assert pairs[("a", "b")]["eval_set"] == "all"
    assert pairs[("a", "b")]["eval_flows"] == plans["b"]["stats"]["all"]["flows"]
    assert pairs[("a", "b")]["shared_groups"] == ["dos"]
    assert pairs[("a", "b")]["unseen_groups"] == ["web"]
    assert pairs[("a", "c")]["shared_groups"] == ["dos"]
    assert pairs[("a", "c")]["unseen_groups"] == []


def test_target_only_cannot_be_a_source(tmp_path):
    c = make_prepared(tmp_path, "c", {"dos": range(0, 3)}, n_blocks=3)
    plans = {"c": plan_target_only(compute_segment_stats(c))}
    with pytest.raises(ValueError):
        build_protocol(plans, sources=("c",))

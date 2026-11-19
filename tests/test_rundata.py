import numpy as np

from nidsrl.data.scaling import load_scaler
from nidsrl.rl.rundata import RUN_MS, load_run, run_ids


def test_runs_cover_each_split_exactly(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    plan = plans["a"]
    for split in ("train", "val", "test"):
        total = 0
        for rid in run_ids(root / "a", plan, split):
            run = load_run(root / "a", plan, split, rid, scaler)
            total += int(run.g_n.sum())
            assert run.n_windows == 180
        assert total == plan["stats"][split]["flows"], split


def test_all_split_includes_purged_rows(world):
    root, plans, _ = world
    plan = plans["a"]
    ids = run_ids(root / "a", plan, "all")
    total = sum(int(load_run(root / "a", plan, "all", r, None).g_n.sum()) for r in ids)
    assert total == plan["stats"]["all"]["flows"]


def test_target_only_runs_are_all_test(world):
    root, plans, _ = world
    plan = plans["c"]
    ids = run_ids(root / "c", plan, "test")
    total = sum(int(load_run(root / "c", plan, "test", r, None).g_n.sum()) for r in ids)
    assert total == plan["stats"]["test"]["flows"]
    assert all(r * RUN_MS >= 0 for r in ids)


def test_features_are_scaled_and_labels_kept(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    plan = plans["a"]
    rid = run_ids(root / "a", plan, "train")[0]
    run = load_run(root / "a", plan, "train", rid, scaler)
    assert run.g_mean.shape[1] == 55 and run.g_max.shape[1] == 35
    assert np.isfinite(run.g_mean).all()
    assert (run.g_A + run.g_B == run.g_n).all()
    assert (run.g_cats.sum(axis=1) == run.g_n).all()

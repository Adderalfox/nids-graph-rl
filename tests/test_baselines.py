import json

import numpy as np
import pytest

from nidsrl.eval.metrics import threshold_at_fpr
from nidsrl.eval.report import collect_results, matrix_markdown, summarize

pytest.importorskip("sklearn")

from nidsrl.experiments.baseline import run_baseline  # noqa: E402
from nidsrl.models.baselines import BASELINES, make_baseline  # noqa: E402


def test_threshold_at_fpr_hits_target():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(20000), np.ones(2000)]).astype(int)
    s = np.concatenate([rng.normal(0, 1, 20000), rng.normal(2, 1, 2000)])
    thr = threshold_at_fpr(y, s, 0.01)
    fpr = float(np.mean(s[y == 0] > thr))
    assert abs(fpr - 0.01) < 0.002
    with pytest.raises(ValueError):
        threshold_at_fpr(np.ones(5), np.arange(5.0))


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        make_baseline("nope", 0)


@pytest.mark.parametrize("model", ["logreg", "hgb"])
def test_baseline_end_to_end_on_synthetic_world(world, model):
    root, _, _ = world
    res = run_baseline(
        root, "a", model, seed=0, train_cap=400, val_cap=200, min_group_flows=50, log=lambda _: None
    )
    pairs = {p["target"]: p for p in res["pairs"]}
    assert set(pairs) == {"a", "b", "c"}
    assert pairs["a"]["kind"] == "in_dataset" and pairs["a"]["eval_set"] == "test"
    # the synthetic first feature separates the classes, so every baseline should find it
    for p in pairs.values():
        assert p["ranking"]["roc_auc"] > 0.99
        assert p["binary"]["tpr"] > 0.9
        assert p["binary"]["fpr"] < 0.05
    assert res["meta"]["model"] == model and res["meta"]["train_rows"] > 0


def test_row_stride_thins_evaluation(world):
    root, _, _ = world
    kw = {"train_cap": 300, "val_cap": 100, "min_group_flows": 50, "log": lambda _: None}
    full = run_baseline(root, "a", "logreg", 0, targets=["b"], **kw)
    thin = run_baseline(root, "a", "logreg", 0, targets=["b"], row_stride=4, **kw)
    assert thin["pairs"][0]["n_flows"] < full["pairs"][0]["n_flows"] / 3


def test_baseline_registry_complete():
    assert set(BASELINES) == {"logreg", "rf", "hgb"}


def test_report_aggregates_over_seeds(world, tmp_path):
    root, _, _ = world
    runs = tmp_path / "runs"
    for seed in (0, 1):
        res = run_baseline(
            root,
            "a",
            "logreg",
            seed,
            train_cap=300,
            val_cap=100,
            min_group_flows=50,
            log=lambda _: None,
        )
        d = runs / f"r{seed}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    df = collect_results(runs, kind="baseline")
    assert set(df["seed"]) == {0, 1} and set(df["target"]) == {"a", "b", "c"}
    summary = summarize(df)
    assert (summary["n_seeds"] == 2).all()
    md = matrix_markdown(summary, "logreg", "roc_auc")
    assert "| a |" in md and "±" in md
    assert collect_results(runs, kind="rl").empty


def test_feature_group_ablation_is_recorded_and_runs(world):
    root, _, _ = world
    res = run_baseline(
        root,
        "a",
        "logreg",
        0,
        train_cap=300,
        val_cap=100,
        min_group_flows=50,
        drop_groups=("host_fingerprint", "retrans"),
        targets=["a"],
        log=lambda _: None,
    )
    assert res["meta"]["n_features"] == 47
    assert res["meta"]["variant"] == "no_host_fingerprint+retrans"
    assert res["pairs"][0]["ranking"]["roc_auc"] > 0.99  # synthetic signal is in log_in_bytes


def test_report_labels_variants_separately(world, tmp_path):
    root, _, _ = world
    runs = tmp_path / "runs"
    for i, drop in enumerate([(), ("host_fingerprint",)]):
        res = run_baseline(
            root,
            "a",
            "logreg",
            0,
            train_cap=300,
            val_cap=100,
            min_group_flows=50,
            drop_groups=drop,
            targets=["a"],
            log=lambda _: None,
        )
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    df = collect_results(runs)
    assert set(df["model"]) == {"logreg", "logreg-no_host_fingerprint"}


def test_keep_only_group_runs_and_is_labelled(world):
    root, _, _ = world
    res = run_baseline(
        root,
        "a",
        "logreg",
        0,
        train_cap=300,
        val_cap=100,
        min_group_flows=50,
        keep_groups=("volume",),
        targets=["a"],
        log=lambda _: None,
    )
    assert res["meta"]["n_features"] == 6
    assert res["meta"]["variant"] == "only_volume"
    assert res["pairs"][0]["ranking"]["roc_auc"] > 0.99  # log_in_bytes is in volume

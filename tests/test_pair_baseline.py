import json

import numpy as np

from nidsrl.eval.report import collect_results, summarize
from nidsrl.experiments.pair_baseline import run_pair_baseline, weighted_threshold_at_fpr


def test_weighted_threshold_matches_weighted_fpr():
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    w = np.array([1.0, 1.0, 1.0, 1.0, 6.0])  # 60% of benign mass sits at score 0.9
    t = weighted_threshold_at_fpr(score, w, 0.5)
    assert t == 0.9  # the heavy row alone exceeds a 50% budget, so nothing may be flagged
    t2 = weighted_threshold_at_fpr(score, w, 0.7)
    above = w[score > t2].sum() / w.sum()
    assert above <= 0.7
    # zero-weight (attack-only) rows are ignored
    assert weighted_threshold_at_fpr(np.array([0.1, 5.0]), np.array([1.0, 0.0]), 0.1) == 0.1


def test_pair_baseline_runs_end_to_end_and_feeds_report(world, tmp_path):
    root, plans, pairs = world
    out = run_pair_baseline(
        root, "a", 0, min_group_flows=50, benign_rate=0.5, log=lambda *_: None
    )
    assert set(out) == {"inspect", "block"}
    for name, res in out.items():
        assert res["meta"]["kind"] == "rl_env" and res["meta"]["model"] == f"hgb-pair-{name}"
        pa = {p["target"]: p for p in res["pairs"]}
        assert set(pa) == {"a", "b", "c"}
        assert pa["a"]["binary"]["tpr"] > 0.8 and pa["a"]["binary"]["fpr"] < 0.2
        assert res["val"]["binary"]["tpr"] > 0.8
    # block prevents more attack flows than a 1-window inspect
    prev = {k: v["pairs"][0]["response"]["prevented_attack_share"] for k, v in out.items()}
    assert prev["block"] > prev["inspect"]

    runs = tmp_path / "runs"
    for i, res in enumerate(out.values()):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(res), encoding="utf-8")
    df = collect_results(runs, kind="rl_env")
    s = summarize(df)
    assert {"prevented_mean", "collateral_mean"} <= set(s.columns)
    assert set(s["model"]) == {"hgb-pair-inspect", "hgb-pair-block"}


def test_threshold_floor_prevents_alert_everything_on_separable_source(world):
    root, _, _ = world
    out = run_pair_baseline(root, "a", 0, min_group_flows=50, log=lambda *_: None)
    m = out["block"]["meta"]
    q = m["threshold_fpr_quantile"]
    assert m["threshold"] == (q if q >= m["saturated_below"] else m["fallback_threshold"])

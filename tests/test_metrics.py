import numpy as np
import pytest

from nidsrl.data.taxonomy import GROUPS
from nidsrl.eval.metrics import (
    binary_report,
    evaluate_arrays,
    group_report,
    ranking_report,
)


def test_binary_report_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    r = binary_report(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (3, 1, 1, 5)
    assert r["tpr"] == pytest.approx(0.75)
    assert r["fpr"] == pytest.approx(1 / 6)
    assert r["precision"] == pytest.approx(0.75)
    assert r["f1_attack"] == pytest.approx(0.75)
    assert r["macro_f1"] == pytest.approx((0.75 + 10 / 12) / 2)
    assert r["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_binary_report_degenerate_cases_return_none():
    r = binary_report(np.zeros(5), np.zeros(5))  # no attacks at all
    assert r["tpr"] is None and r["precision"] is None and r["fpr"] == 0.0


def test_ranking_perfect_and_constant_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = ranking_report(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert perfect["roc_auc"] == pytest.approx(1.0)
    assert perfect["pr_auc"] == pytest.approx(1.0)
    assert perfect["tpr_at_fpr_0.01"] == pytest.approx(1.0)
    const = ranking_report(y, np.zeros(6))
    assert const["roc_auc"] == pytest.approx(0.5)


def test_ranking_single_class_is_none():
    r = ranking_report(np.ones(5), np.arange(5.0))
    assert r["roc_auc"] is None and r["pr_auc"] is None and r["prevalence"] == 1.0


def test_ranking_matches_sklearn_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.2).astype(int)
    s = np.round(rng.normal(size=5000) + 1.2 * y, 1)  # rounding creates many ties
    mine = ranking_report(y, s)
    assert mine["roc_auc"] == pytest.approx(metrics.roc_auc_score(y, s), abs=1e-9)
    assert mine["pr_auc"] == pytest.approx(metrics.average_precision_score(y, s), abs=1e-9)
    fpr, tpr, _ = metrics.roc_curve(y, s)
    ref = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    assert mine["tpr_at_fpr_0.01"] == pytest.approx(ref, abs=1e-9)


def test_group_report_shared_unseen_and_min_flows():
    n = 3000
    group = np.zeros(n, dtype=np.int8)  # benign
    group[:1000] = GROUPS.index("dos")
    group[1000:1600] = GROUPS.index("web")
    group[1600:1650] = GROUPS.index("recon")  # only 50 flows -> excluded
    pred = np.zeros(n, dtype=np.uint8)
    pred[:800] = 1  # 80% of dos detected
    pred[1000:1300] = 1  # 50% of web detected
    r = group_report(pred, group, shared=["dos", "recon"], unseen=["web"], min_group_flows=500)
    assert r["shared"] == {"dos": {"n": 1000, "recall": pytest.approx(0.8)}}
    assert r["unseen"]["web"]["recall"] == pytest.approx(0.5)
    assert r["shared_macro_recall"] == pytest.approx(0.8)
    assert r["unseen_detection_rate"] == pytest.approx(0.5)
    assert r["unseen_flows"] == 600


def test_evaluate_arrays_combines_sections():
    y = np.array([0] * 50 + [1] * 50)
    group = np.array([0] * 50 + [GROUPS.index("dos")] * 50, dtype=np.int8)
    rep = evaluate_arrays(
        y, y, group, y_score=y.astype(float), shared=["dos"], unseen=[], min_group_flows=10
    )
    assert rep["binary"]["macro_f1"] == pytest.approx(1.0)
    assert rep["ranking"]["roc_auc"] == pytest.approx(1.0)
    assert rep["groups"]["shared"]["dos"]["recall"] == 1.0
    assert evaluate_arrays(y, y, group)["ranking"] is None


def test_weighted_ranking_equals_expanded_flows():
    from nidsrl.eval.metrics import ranking_report_weighted

    rng = np.random.default_rng(0)
    score = np.round(rng.normal(size=300), 1)
    pos = rng.integers(0, 6, 300)
    neg = rng.integers(0, 9, 300)
    weighted = ranking_report_weighted(score, pos, neg)
    y = np.concatenate([np.ones(p) for p in pos] + [np.zeros(n) for n in neg])
    s = np.concatenate(
        [np.repeat(score[i], pos[i]) for i in range(300)]
        + [np.repeat(score[i], neg[i]) for i in range(300)]
    )
    expanded = ranking_report(y, s)
    for k in ("roc_auc", "pr_auc", "tpr_at_fpr_0.01"):
        assert weighted[k] == pytest.approx(expanded[k], abs=1e-9)


def test_group_report_weighted():
    from nidsrl.eval.metrics import group_report_weighted

    cats = np.zeros((4, len(GROUPS)))
    cats[:, GROUPS.index("dos")] = [600, 400, 0, 0]
    cats[:, GROUPS.index("web")] = [0, 0, 500, 500]
    alert = np.array([True, False, True, False])
    r = group_report_weighted(alert, cats, ["dos"], ["web"], min_group_flows=100)
    assert r["shared"]["dos"]["recall"] == pytest.approx(0.6)
    assert r["unseen"]["web"]["recall"] == pytest.approx(0.5)
    assert r["unseen_detection_rate"] == pytest.approx(0.5)

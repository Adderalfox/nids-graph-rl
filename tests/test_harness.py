import numpy as np
import pytest

from nidsrl.data.loading import iter_split
from nidsrl.eval.harness import evaluate_pair, save_results


def test_iter_split_matches_plan_statistics_and_never_mixes_splits(world):
    root, plans, _ = world
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = sum(len(b["y"]) for b in iter_split(root / "a", plans["a"], split))
        assert counts[split] == plans["a"]["stats"][split]["flows"]
    everything = sum(len(b["y"]) for b in iter_split(root / "a", plans["a"], "all"))
    assert everything == plans["a"]["stats"]["all"]["flows"]
    assert everything > sum(counts.values())  # purged rows exist only in 'all'


def test_iter_split_rejects_unknown_split(world):
    root, plans, _ = world
    with pytest.raises(ValueError):
        next(iter_split(root / "a", plans["a"], "bogus"))


def test_train_split_is_standardised_with_source_scaler(world):
    root, plans, _ = world
    from nidsrl.data.scaling import load_scaler

    scaler = load_scaler(root / "scalers" / "a.json")
    x = np.concatenate([b["X"] for b in iter_split(root / "a", plans["a"], "train", scaler)])
    assert abs(float(x[:, 0].mean())) < 0.05 and abs(float(x[:, 0].std()) - 1) < 0.05


def _threshold_predictor(scaler):
    thr = (2.5 - scaler["mean"][0]) / scaler["std"][0]

    def predict(x):
        return (x[:, 0] > thr).astype(np.uint8), x[:, 0]

    return predict


def test_oracle_predictor_scores_perfectly_in_dataset_and_transfer(world):
    from nidsrl.data.scaling import load_scaler

    root, _, pairs = world
    predict = _threshold_predictor(load_scaler(root / "scalers" / "a.json"))
    # source a -> target b uses scaler a; both datasets encode the label the same way,
    # so the a-scaled threshold still separates classes.
    for key in (("a", "a"), ("a", "b"), ("a", "c")):
        rep = evaluate_pair(predict, root, pairs[key], min_group_flows=50)
        assert rep["binary"]["tpr"] == pytest.approx(1.0)
        assert rep["binary"]["fpr"] == pytest.approx(0.0)
        assert rep["ranking"]["roc_auc"] == pytest.approx(1.0)
        assert rep["flows_per_second"] > 0


def test_evaluated_flows_follow_protocol_eval_set(world):
    root, plans, pairs = world

    def always_benign(x):
        return np.zeros(len(x), dtype=np.uint8), None

    in_ds = evaluate_pair(always_benign, root, pairs[("a", "a")], min_group_flows=50)
    transfer = evaluate_pair(always_benign, root, pairs[("a", "b")], min_group_flows=50)
    assert in_ds["n_flows"] == plans["a"]["stats"]["test"]["flows"]
    assert transfer["n_flows"] == plans["b"]["stats"]["all"]["flows"]
    assert in_ds["binary"]["tpr"] == 0.0 and in_ds["binary"]["fpr"] == 0.0
    assert in_ds["ranking"] is None  # no scores supplied


def test_unseen_group_detection_is_reported_for_transfer(world):
    from nidsrl.data.scaling import load_scaler

    root, _, pairs = world
    # b was trained on dos+web; target a also contains recon, which b never saw
    predict = _threshold_predictor(load_scaler(root / "scalers" / "b.json"))
    rep = evaluate_pair(predict, root, pairs[("b", "a")], min_group_flows=50)
    assert "recon" in rep["groups"]["unseen"]
    assert rep["groups"]["unseen_detection_rate"] == pytest.approx(1.0)
    assert {"dos", "web"} <= set(rep["groups"]["shared"])


def test_results_are_json_serialisable(world, tmp_path):
    root, _, pairs = world

    def always_attack(x):
        return np.ones(len(x), dtype=np.uint8), np.zeros(len(x), dtype=np.float32)

    rep = evaluate_pair(always_attack, root, pairs[("a", "a")], min_group_flows=50)
    save_results([rep], tmp_path / "out" / "results.json")
    assert (tmp_path / "out" / "results.json").exists()

import pytest

from nidsrl.rl.env import BLOCK, EnvConfig
from nidsrl.rl.rollout import NeverPolicy, OraclePolicy, evaluate_pair_rl


def test_never_policy_flags_nothing(world):
    root, plans, pairs = world
    rep = evaluate_pair_rl(NeverPolicy(), root, pairs[("a", "a")], min_group_flows=50)
    assert rep["n_flows"] == plans["a"]["stats"]["test"]["flows"]
    assert rep["binary"]["tpr"] == 0.0 and rep["binary"]["fpr"] == 0.0
    assert rep["response"]["prevented_attack_share"] == 0.0
    assert rep["response"]["n_decisions"] > 0 and rep["response"]["n_block"] == 0


def test_oracle_block_detects_everything_it_can_and_reports_response(world):
    root, plans, pairs = world
    rep = evaluate_pair_rl(
        OraclePolicy("any", BLOCK),
        root,
        pairs[("a", "b")],
        env_cfg=EnvConfig(block_windows=6),
        min_group_flows=50,
    )
    assert rep["n_flows"] == plans["b"]["stats"]["all"]["flows"]  # transfer = whole target
    assert rep["binary"]["tpr"] > 0.99
    assert rep["ranking"]["roc_auc"] > 0.99
    assert 0 < rep["response"]["prevented_attack_share"] <= 1
    assert (
        rep["groups"]["unseen_detection_rate"] is None
        or rep["groups"]["unseen_detection_rate"] > 0.99
    )
    assert rep["response"]["n_block"] > 0


def test_run_stride_thins_evaluation(world):
    root, _, pairs = world
    full = evaluate_pair_rl(NeverPolicy(), root, pairs[("a", "a")], min_group_flows=50)
    thin = evaluate_pair_rl(
        NeverPolicy(), root, pairs[("a", "a")], min_group_flows=50, run_stride=3
    )
    assert thin["runs"] < full["runs"] and thin["n_flows"] < full["n_flows"]


def test_target_only_dataset_can_be_evaluated(world):
    root, plans, pairs = world
    rep = evaluate_pair_rl(NeverPolicy(), root, pairs[("a", "c")], min_group_flows=50)
    assert rep["n_flows"] == plans["c"]["stats"]["all"]["flows"]


def test_oracle_rule_validation():
    with pytest.raises(ValueError):
        OraclePolicy("sometimes")

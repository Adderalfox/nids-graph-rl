import numpy as np
import pytest
from helpers import CAMPAIGNS

from nidsrl.data.loading import GROUP_INDEX
from nidsrl.data.sampling import inclusion_probabilities, sample_split
from nidsrl.data.scaling import load_scaler


def test_caps_are_respected_and_every_category_is_present(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    cap = 300
    s = sample_split(root / "a", plans["a"], "train", scaler, cap, seed=0)
    counts = np.bincount(s["group"], minlength=8)
    assert counts[GROUP_INDEX["benign"]] <= cap * 1.4
    for g in CAMPAIGNS:
        assert 0 < counts[GROUP_INDEX[g]] <= cap * 1.4
    assert s["X"].shape == (len(s["y"]), 55) and s["X"].dtype == np.float32


def test_small_categories_are_kept_in_full(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    huge_cap = 10**9
    s = sample_split(root / "a", plans["a"], "val", scaler, huge_cap, seed=0)
    assert len(s["y"]) == plans["a"]["stats"]["val"]["flows"]


def test_sampling_is_deterministic_and_seed_dependent(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    a1 = sample_split(root / "a", plans["a"], "train", scaler, 200, seed=3)
    a2 = sample_split(root / "a", plans["a"], "train", scaler, 200, seed=3)
    b = sample_split(root / "a", plans["a"], "train", scaler, 200, seed=4)
    assert np.array_equal(a1["X"], a2["X"])
    assert a1["X"].shape != b["X"].shape or not np.array_equal(a1["X"], b["X"])


def test_probabilities_and_invalid_inputs(world):
    root, plans, _ = world
    p = inclusion_probabilities(plans["a"], "train", 100)
    assert p.shape == (8,) and ((0 <= p) & (p <= 1)).all()
    with pytest.raises(ValueError):
        sample_split(root / "a", plans["a"], "all", None, 100, 0)
    with pytest.raises(ValueError):
        sample_split(root / "c", plans["c"], "train", None, 100, 0)  # target-only

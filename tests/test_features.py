import pytest

from nidsrl.data.prep import FEATURE_COLUMNS
from nidsrl.models.features import FEATURE_GROUPS, kept_feature_indices


def test_groups_partition_all_features_exactly_once():
    names = [n for g in FEATURE_GROUPS.values() for n in g]
    assert sorted(names) == sorted(FEATURE_COLUMNS)
    assert len(names) == len(set(names)) == 55


def test_kept_indices():
    assert kept_feature_indices() == list(range(55))
    kept = kept_feature_indices(["host_fingerprint"])
    assert len(kept) == 51
    assert FEATURE_COLUMNS.index("min_ttl") not in kept
    assert len(kept_feature_indices(["host_fingerprint", "timing"])) == 40


def test_unknown_group_raises():
    with pytest.raises(ValueError):
        kept_feature_indices(["nonsense"])


def test_keep_groups_and_combination():
    assert len(kept_feature_indices(keep_groups=["tcp_flags"])) == 16
    both = kept_feature_indices(keep_groups=["volume", "timing"])
    assert len(both) == 17
    assert len(kept_feature_indices(keep_groups=["volume", "timing"], drop_groups=["timing"])) == 6
    with pytest.raises(ValueError):
        kept_feature_indices(keep_groups=["nonsense"])

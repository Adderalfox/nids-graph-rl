import pytest

from nidsrl.config import load_config


def test_defaults():
    cfg = load_config()
    assert cfg.project.seed == 42
    assert cfg.paths.runs == "experiments/runs"


def test_yaml_file_merges_over_defaults(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("project:\n  seed: 7\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.project.seed == 7
    assert cfg.project.device == "auto"  # untouched default survives


def test_dotted_override_is_typed_by_yaml():
    cfg = load_config(overrides=["project.seed=99", "project.deterministic=false"])
    assert cfg.project.seed == 99
    assert cfg.project.deterministic is False


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        load_config(overrides=["project.sead=1"])


def test_bad_override_format_raises():
    with pytest.raises(ValueError):
        load_config(overrides=["project.seed"])

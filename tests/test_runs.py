import yaml

from nidsrl.config import load_config
from nidsrl.utils.runs import create_run_dir


def test_run_dir_saves_resolved_config(tmp_path):
    cfg = load_config(overrides=["project.seed=5"])
    run_dir = create_run_dir(cfg, tag="unit", root=tmp_path)
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["project"]["seed"] == 5
    assert (run_dir / "provenance.yaml").exists()
    assert run_dir.name.endswith("_unit_s5")

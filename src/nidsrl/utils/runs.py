"""Run directories: every experiment writes its resolved config and provenance."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from nidsrl.config import Config


def git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def create_run_dir(cfg: Config, tag: str = "run", root: str | Path | None = None) -> Path:
    """Create experiments/runs/<timestamp>_<tag>_s<seed>/ and save config + provenance."""
    base = Path(root) if root is not None else Path(cfg.paths.runs)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_{tag}_s{cfg.project.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )
    provenance = {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (run_dir / "provenance.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8"
    )
    return run_dir

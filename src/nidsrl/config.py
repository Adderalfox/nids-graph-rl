"""Typed configuration loaded from YAML with dotted command-line overrides.

Design notes:
- Plain dataclasses + PyYAML: no heavy framework, fully typed, easy to test.
- Unknown keys raise immediately, so a typo in a config can never silently
  change an experiment.
- Later phases add sections (data, graph, model, rl, replay) as new dataclasses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "nids-graph-rl"
    seed: int = 42
    device: str = "auto"
    deterministic: bool = True


@dataclass
class PathsConfig:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    runs: str = "experiments/runs"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_override(tree: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must look like 'section.key=value', got: {override!r}")
    dotted, raw = override.split("=", 1)
    value = yaml.safe_load(raw)
    keys = dotted.strip().split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {dotted!r}: {key!r} is not a section")
    node[keys[-1]] = value


def _build(instance: Any, data: dict[str, Any], path: str = "") -> Any:
    valid = set(instance.__dataclass_fields__)
    for key, value in data.items():
        where = f"{path}{key}"
        if key not in valid:
            raise KeyError(f"Unknown config key: {where!r}")
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section {where!r} must be a mapping")
            _build(current, value, path=f"{where}.")
        else:
            setattr(instance, key, value)
    return instance


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load defaults, merge an optional YAML file, then apply dotted overrides."""
    tree = Config().to_dict()
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        tree = _deep_merge(tree, loaded)
    for override in overrides or []:
        _apply_override(tree, override)
    return _build(Config(), tree)

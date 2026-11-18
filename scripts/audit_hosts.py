"""Host-window audit for the per-host response MDP design (Phase 2b-0).

Example:
    uv run python scripts/audit_hosts.py
    uv run python scripts/audit_hosts.py --datasets unsw --window-s 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.data.hostaudit import GRANULARITIES, audit_entities, format_host_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["unsw", "cic", "ton", "bot"])
    parser.add_argument("--window-s", type=int, default=10)
    parser.add_argument("--granularity", choices=sorted(GRANULARITIES), default="host")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    root = Path(cfg.paths.data_processed)
    out_dir = root / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        report = audit_entities(root / name, args.window_s, GRANULARITIES[args.granularity])
        prefix = "hosts" if args.granularity == "host" else f"entities_{args.granularity}"
        (out_dir / f"{prefix}_{name}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(format_host_report(name, report))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

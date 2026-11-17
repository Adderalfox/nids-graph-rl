"""Audit a dataset CSV and write a JSON report.

Example:
    uv run python scripts/audit_data.py --file data/raw/NF-UNSW-NB15-v3.csv --name unsw_v3
    uv run python scripts/audit_data.py --file data/raw/NF-UNSW-NB15-v3.csv --name unsw_v3 \\
        --max-rows 500000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.data.audit import audit_csv, format_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, help="path to the dataset CSV (.csv/.gz/.zip)")
    parser.add_argument("--name", required=True, help="short dataset name used for the output file")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None, help="audit only the first N rows")
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 60], help="window sizes (s)")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    out_dir = Path(cfg.paths.data_processed) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = audit_csv(
        args.file,
        chunksize=args.chunksize,
        window_seconds=tuple(args.windows),
        max_rows=args.max_rows,
    )
    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Clean a raw NetFlow-v3 CSV into a prepared Parquet dataset.

Example:
    uv run python scripts/prepare_data.py --raw data/raw/NF-UNSW-NB15-v3.csv --name unsw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.data.prep import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="path to the raw dataset CSV")
    parser.add_argument("--name", required=True, help="short name: unsw | cic | ton | bot")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true", help="rebuild if output exists")
    parser.add_argument("--keep-tmp", action="store_true", help="keep temporary block parts")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    out_dir = Path(cfg.paths.data_processed) / args.name
    report = prepare_dataset(
        args.raw,
        out_dir,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
        keep_tmp=args.keep_tmp,
    )
    summary = {k: v for k, v in report.items() if k not in {"features", "dropped_columns"}}
    print(json.dumps(summary, indent=2))
    print(f"\nwritten to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

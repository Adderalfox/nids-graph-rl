"""Plan stratified purged splits, fit source scalers and build the cross-dataset protocol.

Example:
    uv run python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.data.protocol import build_protocol
from nidsrl.data.scaling import fit_scaler, save_scaler
from nidsrl.data.splits import (
    DEFAULT_SEG_MS,
    compute_segment_stats,
    plan_split,
    plan_target_only,
    save_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["unsw", "cic", "ton", "bot"])
    parser.add_argument("--sources", nargs="+", default=["unsw", "cic", "ton"])
    parser.add_argument("--seg-ms", type=int, default=DEFAULT_SEG_MS)
    parser.add_argument("--run-segments", type=int, default=6)
    parser.add_argument("--purge-segments", type=int, default=1)
    parser.add_argument("--min-group-flows", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    root = Path(cfg.paths.data_processed)
    plans = {}

    for name in args.datasets:
        stats = compute_segment_stats(root / name, args.seg_ms)
        if name in args.sources:
            plan = plan_split(
                stats, run_segments=args.run_segments, purge_segments=args.purge_segments
            )
            save_scaler(fit_scaler(root / name, plan), root / "scalers" / f"{name}.json")
        else:
            plan = plan_target_only(stats)
        save_plan(plan, root / "splits" / f"{name}.json")
        plans[name] = plan

        print(f"\n== {name} ({plan['role']}) ==")
        for split in ("train", "val", "test"):
            s = plan["stats"][split]
            print(
                f"  {split:<5} segs={s['segments']:>5} flows={s['flows']:>10,} "
                f"attack_rate={s['attack_rate']:.3f} groups={s['groups']}"
            )
        print(f"  purged segments: {len(plan['purged_segments'])}")
        for w in plan["warnings"]:
            print(f"  WARNING: {w}")

    protocol = build_protocol(plans, tuple(args.sources), args.min_group_flows)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n== protocol (source -> target, eval set) ==")
    for p in protocol["pairs"]:
        print(
            f"  {p['source']:>4} -> {p['target']:<4} {p['eval_set']:<4} "
            f"shared={p['shared_groups']} unseen={p['unseen_groups']}"
        )
    print(f"\nwritten to: {root}/splits, {root}/scalers, {root}/protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

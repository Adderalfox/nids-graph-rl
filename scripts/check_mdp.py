"""Sanity-check the host-pair environment with do-nothing and oracle policies.

The oracle 'any' + block policy follows the same rule as the TIMED BLOCK 60s line of
`scripts/audit_hosts.py --granularity pair`, so prevented/collateral shares should land
close to those audit numbers (small differences: runs are 30 minutes, purged segments
are removed, holds are cut at run boundaries).

Example:
    uv run python scripts/check_mdp.py --dataset unsw
    uv run python scripts/check_mdp.py --dataset ton --run-stride 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nidsrl.config import load_config
from nidsrl.data.splits import load_plan
from nidsrl.rl.env import BLOCK, INSPECT
from nidsrl.rl.rollout import NeverPolicy, OraclePolicy, evaluate_pair_rl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scaler-from", default=None, help="source whose scaler to use")
    parser.add_argument("--eval-set", choices=["test", "all"], default="all")
    parser.add_argument("--run-stride", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    root = Path(cfg.paths.data_processed)
    plan = load_plan(root / "splits" / f"{args.dataset}.json")
    scaler_from = args.scaler_from or (args.dataset if plan["role"] == "source" else "ton")
    pair = {
        "source": scaler_from,
        "target": args.dataset,
        "kind": "diagnostic",
        "eval_set": args.eval_set,
        "shared_groups": [],
        "unseen_groups": [],
    }

    policies = {
        "never (monitor only)": NeverPolicy(),
        "oracle any-attack -> block 60s": OraclePolicy("any", BLOCK),
        "oracle any-attack -> inspect 10s": OraclePolicy("any", INSPECT),
        "oracle majority -> block 60s": OraclePolicy("majority", BLOCK),
    }
    print(f"dataset={args.dataset} eval_set={args.eval_set} scaler={scaler_from}")
    for name, policy in policies.items():
        rep = evaluate_pair_rl(policy, root, pair, run_stride=args.run_stride)
        b, r = rep["binary"], rep["response"]

        def pct(x: float | None) -> str:
            return "-" if x is None else f"{100 * x:5.1f}%"

        print(
            f"  {name:<34} TPR {pct(b['tpr'])} FPR {pct(b['fpr'])} macroF1 {pct(b['macro_f1'])} | "
            f"prevented {pct(r['prevented_attack_share'])} collateral "
            f"{pct(r['collateral_benign_share'])} | blocks {r['n_block']:,} "
            f"inspects {r['n_inspect']:,} ({rep['runs']} runs, {rep['seconds']:.0f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

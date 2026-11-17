"""Cross-dataset evaluation protocol.

Rules (fixed before any model is trained):
1. Source datasets (unsw, cic, ton) train and validate on their own train/val rows.
   Hyperparameters and early stopping use ONLY source validation data.
2. In-dataset reference (source == target): evaluated on the source's own TEST rows.
3. Zero-shot transfer (source != target): evaluated on the WHOLE target dataset
   (every row, nothing from the target is used in training or tuning). BoT-IoT is
   target-only and is always evaluated on all rows.
4. The scaler is fit on source train rows only and applied unchanged to the target.
5. Per-group recall is reported only for groups seen in source training ("shared").
   Groups in the evaluation set that are absent from source training ("unseen")
   get a separate detection-rate number.
"""

from __future__ import annotations

from typing import Any

SOURCES_DEFAULT = ("unsw", "cic", "ton")


def build_protocol(
    plans: dict[str, dict[str, Any]],
    sources: tuple[str, ...] = SOURCES_DEFAULT,
    min_group_flows: int = 500,
) -> dict[str, Any]:
    """Build the list of (source, target) evaluations with shared/unseen attack groups."""
    pairs = []
    for s in sources:
        if s not in plans:
            raise KeyError(f"Source dataset {s!r} has no plan")
        if plans[s]["role"] != "source":
            raise ValueError(f"{s!r} is target-only and cannot be a training source")
        src_groups = {
            g for g, c in plans[s]["stats"]["train"]["groups"].items() if c >= min_group_flows
        }
        for t, plan in plans.items():
            eval_set = "test" if s == t else "all"
            eval_stats = plan["stats"][eval_set]
            tgt_groups = {g for g, c in eval_stats["groups"].items() if c >= min_group_flows}
            pairs.append(
                {
                    "source": s,
                    "target": t,
                    "kind": "in_dataset" if s == t else "transfer",
                    "eval_set": eval_set,
                    "eval_flows": eval_stats["flows"],
                    "eval_attack_rate": eval_stats["attack_rate"],
                    "shared_groups": sorted(src_groups & tgt_groups),
                    "unseen_groups": sorted(tgt_groups - src_groups),
                }
            )
    return {
        "sources": list(sources),
        "targets": sorted(plans),
        "min_group_flows": min_group_flows,
        "pairs": pairs,
    }

"""Host-window audit: is a per-host response MDP realistic on these datasets?

For a per-host-per-window agent (monitor / inspect / block) to make sense, the data
must show that:
- the number of active source hosts per window is manageable;
- most host-windows are pure (all benign or all attack), so a host-level label is
  meaningful, and mixed hosts are a small, quantifiable ambiguity;
- attacker hosts persist across windows, otherwise blocking after detection prevents
  nothing (one-shot or spoofed sources);
- blocking an attacker costs little benign traffic (collateral).

All statistics are computed per hourly block (windows never straddle blocks), so a
host that persists across an hour boundary is counted as new in the next block; this
slightly understates persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidsrl.eval.metrics import binary_report


def _dist(values: list[np.ndarray]) -> dict[str, float | None]:
    arr = np.concatenate(values) if values else np.array([])
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


HOST = ("src_id",)
PAIR = ("src_id", "dst_id")
SERVICE = ("src_id", "dst_id", "dst_port")
GRANULARITIES = {"host": HOST, "pair": PAIR, "service": SERVICE}


def audit_hosts(dataset_dir: str | Path, window_s: int = 10) -> dict[str, Any]:
    """Host-level audit (kept for backward compatibility)."""
    return audit_entities(dataset_dir, window_s, HOST)


BLOCK_WINDOWS_DEFAULT = (1, 3, 6, 30)


def _timed_block_sim(g: pd.DataFrame, ks: tuple[int, ...]) -> dict[int, tuple[int, int, int]]:
    """Oracle timed-block simulation for one block.

    g has one row per (window w, entity e) with attack flows `a` and benign flows `b`.
    Oracle policy: at an entity's attack window, if not currently blocked, block the
    entity for the next k windows. Returns {k: (prevented_attack, collateral_benign, blocks)}.
    """
    g = g.sort_values(["e", "w"], kind="stable").reset_index(drop=True)
    e = g["e"].to_numpy(np.int64)
    w = (g["w"] - g["w"].min()).to_numpy(np.int64)
    a = g["a"].to_numpy(np.int64)
    b = g["b"].to_numpy(np.int64)
    key = e * (1 << 20) + w
    ca = np.concatenate([[0], np.cumsum(a)])
    cb = np.concatenate([[0], np.cumsum(b)])
    attack_rows = np.nonzero(a > 0)[0].tolist()
    e_l, w_l = e.tolist(), w.tolist()
    out = {}
    for k in ks:
        end_row = np.searchsorted(key, key + k, side="right").tolist()
        prevented = collateral = blocks = 0
        last_e, until = -1, -1
        for i in attack_rows:
            if e_l[i] != last_e:
                last_e, until = e_l[i], -1
            if w_l[i] <= until:
                continue
            until = w_l[i] + k
            j = end_row[i]
            prevented += int(ca[j] - ca[i + 1])
            collateral += int(cb[j] - cb[i + 1])
            blocks += 1
        out[k] = (prevented, collateral, blocks)
    return out


def audit_entities(
    dataset_dir: str | Path,
    window_s: int = 10,
    keys: tuple[str, ...] = HOST,
    block_windows: tuple[int, ...] = BLOCK_WINDOWS_DEFAULT,
) -> dict[str, Any]:
    """Entity-window statistics, where an entity is the tuple of `keys`.

    keys=("src_id",) audits per-host decisions; ("src_id", "dst_id") per host pair
    (a firewall-rule-like granularity); ("src_id", "dst_id", "dst_port") per service.
    """
    win_ms = window_s * 1000
    hosts_per_window: list[np.ndarray] = []
    attackers_per_window: list[np.ndarray] = []
    windows_per_attacker: list[np.ndarray] = []

    n_hw = pure_attack_hw = pure_benign_hw = mixed_hw = 0
    flows_pure_attack = flows_pure_benign = flows_mixed = 0
    mixed_attack_share: list[np.ndarray] = []
    total_attack = preventable = 0
    total_benign = benign_from_attackers_after_first = 0
    attacker_hosts = one_shot_attackers = 0
    timed = {k: [0, 0, 0] for k in block_windows}
    maj = [0, 0, 0, 0]  # tp, fn, fp, tn under the majority-label oracle
    anyr = [0, 0, 0, 0]  # same under the any-attack-blocks oracle

    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        df = pd.read_parquet(path, columns=["start_ms", *keys, "label"])
        df["e"] = df.groupby(list(keys)).ngroup()
        df["w"] = (df["start_ms"].to_numpy() // win_ms).astype(np.int64)
        g = (
            df.groupby(["w", "e"])["label"]
            .agg(n="size", a="sum")
            .reset_index()
            .astype({"n": np.int64, "a": np.int64})
        )
        g["b"] = g["n"] - g["a"]

        a_arr, b_arr = g["a"].to_numpy(np.int64), g["b"].to_numpy(np.int64)
        for rule, flag in (
            (maj, (a_arr >= b_arr) & (a_arr > 0)),
            (anyr, a_arr > 0),
        ):
            rule[0] += int(a_arr[flag].sum())
            rule[1] += int(a_arr[~flag].sum())
            rule[2] += int(b_arr[flag].sum())
            rule[3] += int(b_arr[~flag].sum())

        hosts_per_window.append(g.groupby("w").size().to_numpy())
        n_hw += len(g)
        pa = g["a"] == g["n"]
        pb = g["a"] == 0
        mx = ~(pa | pb)
        pure_attack_hw += int(pa.sum())
        pure_benign_hw += int(pb.sum())
        mixed_hw += int(mx.sum())
        flows_pure_attack += int(g.loc[pa, "n"].sum())
        flows_pure_benign += int(g.loc[pb, "n"].sum())
        flows_mixed += int(g.loc[mx, "n"].sum())
        if mx.any():
            mixed_attack_share.append((g.loc[mx, "a"] / g.loc[mx, "n"]).to_numpy())

        total_benign += int(g["b"].sum())
        ah = g[g["a"] > 0]
        if not ah.empty:
            for k, (pv, co, bl) in _timed_block_sim(g, block_windows).items():
                timed[k][0] += pv
                timed[k][1] += co
                timed[k][2] += bl
        if ah.empty:
            continue
        total_attack += int(ah["a"].sum())
        attackers_per_window.append(ah.groupby("w")["e"].nunique().to_numpy())
        per_host_windows = ah.groupby("e")["w"].nunique()
        windows_per_attacker.append(per_host_windows.to_numpy())
        attacker_hosts += len(per_host_windows)
        one_shot_attackers += int((per_host_windows == 1).sum())

        first = ah.groupby("e")["w"].min()
        ah_first = ah["e"].map(first)
        preventable += int(ah.loc[ah["w"] > ah_first, "a"].sum())

        mine = g[g["e"].isin(first.index)]
        mine_first = mine["e"].map(first)
        benign_from_attackers_after_first += int(mine.loc[mine["w"] > mine_first, "b"].sum())

    total_flows = total_attack + total_benign
    all_hw_flows = flows_pure_attack + flows_pure_benign + flows_mixed
    return {
        "window_s": window_s,
        "entity": "+".join(keys),
        "flows": int(total_flows),
        "host_windows": int(n_hw),
        "active_src_hosts_per_window": _dist(hosts_per_window),
        "attacker_hosts_per_attack_window": _dist(attackers_per_window),
        "host_window_purity": {
            "pure_benign_share": pure_benign_hw / n_hw if n_hw else None,
            "pure_attack_share": pure_attack_hw / n_hw if n_hw else None,
            "mixed_share": mixed_hw / n_hw if n_hw else None,
            "flows_in_mixed_share": flows_mixed / all_hw_flows if all_hw_flows else None,
            "mixed_attack_fraction": _dist(mixed_attack_share),
        },
        "attacker_persistence": {
            "attacker_host_blocks": int(attacker_hosts),
            "one_shot_share": (one_shot_attackers / attacker_hosts) if attacker_hosts else None,
            "windows_per_attacker": _dist(windows_per_attacker),
            "preventable_attack_flow_share": (preventable / total_attack) if total_attack else None,
        },
        "flow_level_ceiling": {
            "majority_rule": _ceiling(maj),
            "any_attack_rule": _ceiling(anyr),
        },
        "timed_block": {
            str(k): {
                "seconds": k * window_s,
                "prevented_attack_share": (v[0] / total_attack) if total_attack else None,
                "collateral_benign_share": (v[1] / total_benign) if total_benign else None,
                "blocks": v[2],
            }
            for k, v in timed.items()
        },
        "collateral": {
            "benign_flows_from_attackers_after_first_window": int(
                benign_from_attackers_after_first
            ),
            "share_of_all_benign_flows": (
                benign_from_attackers_after_first / total_benign if total_benign else None
            ),
        },
    }


def _ceiling(c: list[int]) -> dict[str, float | None]:
    """Best flow-level metrics when every flow inherits its entity-window's oracle label."""
    tp, fn, fp, tn = c
    y_true = np.concatenate([np.ones(tp + fn, dtype=np.uint8), np.zeros(fp + tn, dtype=np.uint8)])
    y_pred = np.concatenate(
        [
            np.ones(tp, dtype=np.uint8),
            np.zeros(fn, dtype=np.uint8),
            np.ones(fp, dtype=np.uint8),
            np.zeros(tn, dtype=np.uint8),
        ]
    )
    r = binary_report(y_true, y_pred) if len(y_true) else {}
    return {"tpr": r.get("tpr"), "fpr": r.get("fpr"), "macro_f1": r.get("macro_f1")}


def format_host_report(name: str, r: dict[str, Any]) -> str:
    a, p, c = r["active_src_hosts_per_window"], r["host_window_purity"], r["collateral"]
    ap, att = r["attacker_persistence"], r["attacker_hosts_per_attack_window"]

    def f(x: float | None, pct: bool = False) -> str:
        if x is None:
            return "-"
        return f"{100 * x:.1f}%" if pct else f"{x:,.0f}"

    return "\n".join(
        [
            f"== {name} (entity {r.get('entity', 'src_id')}, window {r['window_s']}s) ==",
            f"  active src hosts/window : median {f(a['median'])}  "
            f"p95 {f(a['p95'])}  max {f(a['max'])}",
            f"  attacker hosts/attack window: median {f(att['median'])}  p95 {f(att['p95'])}",
            f"  host-windows            : benign-only {f(p['pure_benign_share'], True)}  "
            f"attack-only {f(p['pure_attack_share'], True)}  mixed {f(p['mixed_share'], True)}",
            f"  flows in mixed hosts    : {f(p['flows_in_mixed_share'], True)}",
            f"  attacker persistence    : one-shot {f(ap['one_shot_share'], True)}  "
            f"windows/attacker median {f(ap['windows_per_attacker']['median'])} "
            f"p95 {f(ap['windows_per_attacker']['p95'])}",
            f"  PREVENTABLE attack flows: {f(ap['preventable_attack_flow_share'], True)} "
            "(flows after an attacker's first attack window)",
            f"  COLLATERAL benign flows : {f(c['share_of_all_benign_flows'], True)} of all benign "
            "(benign flows from attacker hosts after their first attack window)",
            *[
                f"  CEILING ({name_}): TPR {f(c_['tpr'], True)}  FPR {f(c_['fpr'], True)}  "
                f"macroF1 {f(c_['macro_f1'], True)}"
                for name_, c_ in (
                    ("majority label ", r["flow_level_ceiling"]["majority_rule"]),
                    ("any-attack rule ", r["flow_level_ceiling"]["any_attack_rule"]),
                )
            ],
            *[
                f"  TIMED BLOCK {t['seconds']:>4}s      : prevents "
                f"{f(t['prevented_attack_share'], True)} of attack flows, "
                f"loses {f(t['collateral_benign_share'], True)} of benign ({t['blocks']:,} blocks)"
                for t in r.get("timed_block", {}).values()
            ],
        ]
    )

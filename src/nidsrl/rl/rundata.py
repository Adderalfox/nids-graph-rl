"""Load one 30-minute run as entity-window groups for the response environment.

An entity is a (src host, dst host) pair; a group is one entity's flows inside one
10 s window. Per group we keep the mean and max of the scaled flow features, the flow
count, and ground-truth attack/benign flow counts (used for rewards and evaluation only,
never shown to the agent).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidsrl.data.loading import GROUP_INDEX
from nidsrl.data.prep import BLOCK_MS, FEATURE_COLUMNS
from nidsrl.data.scaling import N_CONTINUOUS, apply_scaler
from nidsrl.data.splits import SPLIT_NAMES, assign_rows
from nidsrl.data.taxonomy import GROUPS, OTHER

RUN_MS = 1_800_000  # 30 minutes = 6 segments of 5 minutes
N_FEATURES = len(FEATURE_COLUMNS)


@dataclass
class RunData:
    run_id: int
    n_windows: int
    n_pairs: int
    pair_src: np.ndarray  # (n_pairs,) local host ids
    pair_dst: np.ndarray
    g_win: np.ndarray  # (G,) window index, sorted by (window, pair)
    g_pair: np.ndarray  # (G,)
    g_n: np.ndarray  # (G,) flows
    g_A: np.ndarray  # (G,) attack flows
    g_B: np.ndarray  # (G,) benign flows
    g_cats: np.ndarray  # (G, len(GROUPS)) flows per taxonomy category
    g_mean: np.ndarray  # (G, N_FEATURES)
    g_max: np.ndarray  # (G, N_CONTINUOUS)
    win_ptr: np.ndarray  # (n_windows + 1,) CSR offsets into groups
    order_pw: np.ndarray  # group indices sorted by (pair, window)
    pos_in_pw: np.ndarray  # inverse of order_pw

    @property
    def n_groups(self) -> int:
        return len(self.g_win)


def run_ids(dataset_dir: str | Path, plan: dict[str, Any], split: str) -> list[int]:
    """Run ids for a split ('train'/'val'/'test' from the plan, or 'all' = every run with data)."""
    if plan["role"] == "source" and plan["seg_ms"] * plan["run_segments"] != RUN_MS:
        raise ValueError("Plan run length differs from RUN_MS")
    if split in SPLIT_NAMES and plan["role"] == "source":
        return list(plan["runs"][split])
    ids: set[int] = set()
    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        start = pd.read_parquet(path, columns=["start_ms"])["start_ms"].to_numpy()
        ids.update(np.unique(start // RUN_MS).tolist())
    return sorted(ids)


def _empty(run_id: int, n_windows: int) -> RunData:
    z = np.zeros(0, dtype=np.int64)
    return RunData(
        run_id=run_id,
        n_windows=n_windows,
        n_pairs=0,
        pair_src=np.zeros(0, np.int32),
        pair_dst=np.zeros(0, np.int32),
        g_win=z,
        g_pair=z,
        g_n=z,
        g_A=z,
        g_B=z,
        g_cats=np.zeros((0, len(GROUPS)), np.int64),
        g_mean=np.zeros((0, N_FEATURES), np.float32),
        g_max=np.zeros((0, N_CONTINUOUS), np.float32),
        win_ptr=np.zeros(n_windows + 1, np.int64),
        order_pw=z,
        pos_in_pw=z,
    )


def build_groups(
    run_id: int,
    win: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cat: np.ndarray,
    n_windows: int,
) -> RunData:
    """Aggregate flows into (window, pair) groups."""
    if len(win) == 0:
        return _empty(run_id, n_windows)
    key = (src.astype(np.int64) << 32) | dst.astype(np.int64)
    uniq, pair = np.unique(key, return_inverse=True)
    pair = pair.astype(np.int64)
    n_pairs = len(uniq)

    gkey = win.astype(np.int64) * n_pairs + pair
    order = np.argsort(gkey, kind="stable")
    gk = gkey[order]
    starts = np.flatnonzero(np.concatenate([[True], gk[1:] != gk[:-1]]))
    counts = np.diff(np.concatenate([starts, [len(gk)]]))

    xs = x[order]
    ys = y[order].astype(np.int64)
    cs = cat[order]
    g_A = np.add.reduceat(ys, starts)
    g_cats = np.stack(
        [np.add.reduceat((cs == c).astype(np.int64), starts) for c in range(len(GROUPS))], axis=1
    )
    g_mean = (np.add.reduceat(xs, starts, axis=0) / counts[:, None]).astype(np.float32)
    g_max = np.maximum.reduceat(xs[:, :N_CONTINUOUS], starts, axis=0).astype(np.float32)
    g_win = win[order][starts].astype(np.int64)
    g_pair = pair[order][starts]
    order_pw = np.lexsort((g_win, g_pair))
    pos = np.empty(len(order_pw), dtype=np.int64)
    pos[order_pw] = np.arange(len(order_pw))
    return RunData(
        run_id=run_id,
        n_windows=n_windows,
        n_pairs=n_pairs,
        pair_src=(uniq >> 32).astype(np.int32),
        pair_dst=(uniq & 0xFFFFFFFF).astype(np.int32),
        g_win=g_win,
        g_pair=g_pair,
        g_n=counts.astype(np.int64),
        g_A=g_A,
        g_B=counts.astype(np.int64) - g_A,
        g_cats=g_cats,
        g_mean=g_mean,
        g_max=g_max,
        win_ptr=np.searchsorted(g_win, np.arange(n_windows + 1)).astype(np.int64),
        order_pw=order_pw,
        pos_in_pw=pos,
    )


def load_run(
    dataset_dir: str | Path,
    plan: dict[str, Any],
    split: str,
    run_id: int,
    scaler: dict[str, Any] | None,
    window_s: int = 10,
) -> RunData:
    """Load one run of a split ('train'/'val'/'test'/'all') as entity-window groups."""
    win_ms = window_s * 1000
    n_windows = RUN_MS // win_ms
    start = run_id * RUN_MS
    path = Path(dataset_dir) / "flows" / f"block-{start // BLOCK_MS:09d}.parquet"
    if not path.exists():
        return _empty(run_id, n_windows)
    df = pd.read_parquet(path)
    df = df.loc[(df["start_ms"] >= start) & (df["start_ms"] < start + RUN_MS)]
    if split != "all":
        df = df.loc[assign_rows(df["start_ms"].to_numpy(), plan) == SPLIT_NAMES.index(split)]
    if df.empty:
        return _empty(run_id, n_windows)
    feats = df[list(FEATURE_COLUMNS)]
    if scaler is not None:
        feats = apply_scaler(feats, scaler)
    other = GROUP_INDEX[OTHER]
    cat = df["attack_group"].map(lambda g: GROUP_INDEX.get(g, other)).to_numpy(dtype=np.int8)
    return build_groups(
        run_id,
        ((df["start_ms"].to_numpy() - start) // win_ms).astype(np.int64),
        df["src_id"].to_numpy(),
        df["dst_id"].to_numpy(),
        feats.to_numpy(dtype=np.float32),
        df["label"].to_numpy(dtype=np.uint8),
        cat,
        n_windows,
    )

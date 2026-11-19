"""Shared synthetic-data helpers for tests."""

import numpy as np
import pandas as pd

from nidsrl.data.prep import FEATURE_COLUMNS, META_COLUMNS

BASE_BLOCK = 420_000
HOUR = 3_600_000


def make_prepared(root, name, campaigns, n_blocks=24, flows_per_block=1200, seed=0):
    """Synthetic prepared dataset whose attack groups are scheduled in time, like CIC/ToN.

    campaigns maps group name -> range of block indices in which that group occurs.
    """
    rng = np.random.default_rng(seed)
    flows_dir = root / name / "flows"
    flows_dir.mkdir(parents=True)
    for b in range(n_blocks):
        n = flows_per_block
        start = np.sort((BASE_BLOCK + b) * HOUR + rng.integers(0, HOUR, n))
        group = np.full(n, "benign", dtype=object)
        for g, blocks in campaigns.items():
            if b in blocks:
                group[rng.random(n) < 0.3] = g
        df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS))
        label = (group != "benign").astype(np.uint8)
        # first feature separates classes so simple predictors can be tested
        df["log_in_bytes"] = label * 5.0 + rng.normal(0, 0.1, n)
        df.insert(0, "start_ms", start)
        df.insert(1, "end_ms", start + 5)
        # attacks come from a few attacker hosts, benign traffic from many others, so that
        # host pairs are (almost) pure like in the real datasets
        is_attack = label == 1
        src = np.where(is_attack, rng.integers(0, 4, n), rng.integers(10, 30, n))
        df.insert(2, "src_id", src.astype(np.int32))
        df.insert(3, "dst_id", rng.integers(100, 105, n).astype(np.int32))
        df.insert(4, "src_port", 1)
        df.insert(5, "dst_port", 80)
        df.insert(6, "label", label)
        df.insert(7, "attack", group)
        df.insert(8, "attack_group", group)
        assert list(df.columns) == list(META_COLUMNS) + list(FEATURE_COLUMNS)
        df.to_parquet(flows_dir / f"block-{BASE_BLOCK + b:09d}.parquet", index=False)
    return root / name


CAMPAIGNS = {"dos": range(0, 8), "recon": range(8, 16), "web": range(16, 24)}


def load_all(d):
    return pd.concat([pd.read_parquet(p) for p in sorted((d / "flows").glob("*.parquet"))])

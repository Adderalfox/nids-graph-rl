"""Phase 1b: clean NetFlow-v3 CSVs into a leakage-safe, time-sorted Parquet dataset.

Design:
- Streaming two-pass pipeline (bounded RAM; works on 20M+ row files on a 16 GB machine).
  Pass 1 reads the CSV in chunks, transforms each chunk and appends it to one-hour
  time-block part files. Pass 2 sorts each block by start time and writes the final
  Parquet file. Sorting block-by-block fixes slightly unsorted exports without ever
  loading a whole dataset.
- Blocks are one hour long, so any window size that divides 3600 s never straddles a
  block boundary (10 s and 60 s both qualify).
- Features are identity-free: no IPs, ports or absolute timestamps. Those are kept as
  metadata columns for graph construction only.
- Features are transformed but NOT standardised here: scalers must be fit on training
  data only, which is decided in Phase 1c.
- Hosts get a stable integer id per dataset (first-appearance order). The id is a key
  for temporal state and history features in later phases; it must never be fed to a
  model as a feature or embedding table.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nidsrl.data.taxonomy import coarse_attack, is_known

BLOCK_MS = 3_600_000  # one hour

# Raw column names
SRC_IP, DST_IP = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
SRC_PORT, DST_PORT = "L4_SRC_PORT", "L4_DST_PORT"
START, END = "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
LABEL, ATTACK = "Label", "Attack"
PROTOCOL = "PROTOCOL"
CLIENT_FLAGS, SERVER_FLAGS = "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"

# Dropped on purpose (documented in docs/data.md):
#  *_SECOND_BYTES: computed inconsistently across datasets (inf/NaN in UNSW and BoT)
#  L7_PROTO: categorical id, meaningless as a number and dataset-specific
#  TCP_FLAGS: redundant with the decoded client/server flags
#  ICMP_TYPE: redundant with ICMP_IPV4_TYPE
#  DNS_*: sparse, random ids and wildly different value ranges across datasets
#  FTP_COMMAND_RET_CODE: constant in CIC, rarely populated elsewhere
DROPPED_COLUMNS = (
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "L7_PROTO",
    "TCP_FLAGS",
    "ICMP_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
)

# log1p-transformed (heavy-tailed counts, sizes, durations, rates)
LOG1P_COLUMNS = (
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
)

# fixed-scale columns: divided by their natural maximum (no fitting required)
SCALED_COLUMNS = {
    "MIN_TTL": 255.0,
    "MAX_TTL": 255.0,
    "TCP_WIN_MAX_IN": 65535.0,
    "TCP_WIN_MAX_OUT": 65535.0,
    "ICMP_IPV4_TYPE": 255.0,
}

FLAG_BITS = ("fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr")
PROTO_NAMES = ("tcp", "udp", "icmp", "other")

FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(f"log_{c.lower()}" for c in LOG1P_COLUMNS)
    + tuple(c.lower() for c in SCALED_COLUMNS)
    + tuple(f"client_{b}" for b in FLAG_BITS)
    + tuple(f"server_{b}" for b in FLAG_BITS)
    + tuple(f"proto_{p}" for p in PROTO_NAMES)
)

META_COLUMNS = (
    "start_ms",
    "end_ms",
    "src_id",
    "dst_id",
    "src_port",
    "dst_port",
    "label",
    "attack",
    "attack_group",
)

REQUIRED_RAW = (
    {SRC_IP, DST_IP, SRC_PORT, DST_PORT, START, END, LABEL, ATTACK, PROTOCOL}
    | {CLIENT_FLAGS, SERVER_FLAGS}
    | set(LOG1P_COLUMNS)
    | set(SCALED_COLUMNS)
)


class HostIndex:
    """Stable integer ids for IP strings, assigned in first-appearance order."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def encode(self, ips: pd.Series) -> np.ndarray:
        strs = ips.astype(str)
        for ip in strs.unique():
            if ip not in self._ids:
                self._ids[ip] = len(self._ids)
        return strs.map(self._ids).to_numpy(dtype=np.int32)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"host_id": np.arange(len(self._ids), dtype=np.int32), "ip": list(self._ids)}
        )


def _decode_flags(values: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    v = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
    return {
        f"{prefix}_{name}": ((v >> bit) & 1).astype(np.uint8) for bit, name in enumerate(FLAG_BITS)
    }


def transform_chunk(raw: pd.DataFrame, hosts: HostIndex) -> tuple[pd.DataFrame, dict[str, int]]:
    """Transform one raw chunk. Returns (clean frame, counters)."""
    missing = REQUIRED_RAW - set(raw.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    start = pd.to_numeric(raw[START], errors="coerce")
    end = pd.to_numeric(raw[END], errors="coerce")
    ok = start.notna() & end.notna() & (start > 0)
    dropped_time = int((~ok).sum())
    raw = raw.loc[ok]
    start, end = start[ok], end[ok]

    out: dict[str, Any] = {}
    nonfinite = 0

    for col in LOG1P_COLUMNS:
        x = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        bad = ~np.isfinite(x)
        nonfinite += int(bad.sum())
        x = np.where(bad, 0.0, x)
        out[f"log_{col.lower()}"] = np.log1p(np.clip(x, 0.0, None)).astype(np.float32)

    for col, scale in SCALED_COLUMNS.items():
        x = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        bad = ~np.isfinite(x)
        nonfinite += int(bad.sum())
        x = np.where(bad, 0.0, x)
        out[col.lower()] = np.clip(x / scale, 0.0, 1.0).astype(np.float32)

    out.update(
        _decode_flags(pd.to_numeric(raw[CLIENT_FLAGS], errors="coerce").to_numpy(), "client")
    )
    out.update(
        _decode_flags(pd.to_numeric(raw[SERVER_FLAGS], errors="coerce").to_numpy(), "server")
    )

    proto = pd.to_numeric(raw[PROTOCOL], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    out["proto_tcp"] = (proto == 6).astype(np.uint8)
    out["proto_udp"] = (proto == 17).astype(np.uint8)
    out["proto_icmp"] = np.isin(proto, (1, 58)).astype(np.uint8)
    out["proto_other"] = (1 - out["proto_tcp"] - out["proto_udp"] - out["proto_icmp"]).astype(
        np.uint8
    )

    attack = raw[ATTACK].astype(str)
    frame = pd.DataFrame(out)
    frame.insert(0, "start_ms", start.to_numpy(dtype=np.int64))
    frame.insert(1, "end_ms", end.to_numpy(dtype=np.int64))
    frame.insert(2, "src_id", hosts.encode(raw[SRC_IP]))
    frame.insert(3, "dst_id", hosts.encode(raw[DST_IP]))
    frame.insert(
        4,
        "src_port",
        pd.to_numeric(raw[SRC_PORT], errors="coerce").fillna(0).to_numpy(dtype=np.uint16),
    )
    frame.insert(
        5,
        "dst_port",
        pd.to_numeric(raw[DST_PORT], errors="coerce").fillna(0).to_numpy(dtype=np.uint16),
    )
    frame.insert(
        6, "label", pd.to_numeric(raw[LABEL], errors="coerce").fillna(0).to_numpy(dtype=np.uint8)
    )
    frame.insert(7, "attack", attack.to_numpy())
    frame.insert(8, "attack_group", attack.map(coarse_attack).to_numpy())

    unknown = int((~attack.map(is_known)).sum())
    counters = {
        "dropped_bad_time": dropped_time,
        "nonfinite_replaced": nonfinite,
        "unknown_attack_rows": unknown,
    }
    return frame[list(META_COLUMNS) + list(FEATURE_COLUMNS)], counters


def prepare_dataset(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    chunksize: int = 500_000,
    overwrite: bool = False,
    keep_tmp: bool = False,
) -> dict[str, Any]:
    """Run the full two-pass pipeline and return the preparation report."""
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True to rebuild")
        shutil.rmtree(out_dir)
    tmp_dir = out_dir / "_tmp"
    flows_dir = out_dir / "flows"
    tmp_dir.mkdir(parents=True)
    flows_dir.mkdir(parents=True)

    hosts = HostIndex()
    totals = {
        "rows_in": 0,
        "rows_out": 0,
        "dropped_bad_time": 0,
        "nonfinite_replaced": 0,
        "unknown_attack_rows": 0,
    }
    label_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    blocks_seen: set[int] = set()

    # Pass 1: stream, transform, scatter into hour blocks
    for i, raw in enumerate(pd.read_csv(csv_path, chunksize=chunksize, low_memory=False)):
        totals["rows_in"] += len(raw)
        frame, counters = transform_chunk(raw, hosts)
        for k, v in counters.items():
            totals[k] += v
        totals["rows_out"] += len(frame)
        for k, v in frame["label"].value_counts().items():
            label_counts[str(k)] = label_counts.get(str(k), 0) + int(v)
        for k, v in frame["attack_group"].value_counts().items():
            group_counts[str(k)] = group_counts.get(str(k), 0) + int(v)

        block_ids = frame["start_ms"].to_numpy() // BLOCK_MS
        for block, part in frame.groupby(block_ids, sort=False):
            block = int(block)
            blocks_seen.add(block)
            bdir = tmp_dir / f"b{block:09d}"
            bdir.mkdir(exist_ok=True)
            part.to_parquet(bdir / f"p{i:06d}.parquet", index=False)

    # Pass 2: sort each block and write final files
    for block in sorted(blocks_seen):
        bdir = tmp_dir / f"b{block:09d}"
        df = pd.concat(
            [pd.read_parquet(p) for p in sorted(bdir.glob("*.parquet"))], ignore_index=True
        )
        df = df.sort_values(["start_ms", "end_ms"], kind="stable").reset_index(drop=True)
        df.to_parquet(flows_dir / f"block-{block:09d}.parquet", index=False, compression="zstd")

    hosts.to_frame().to_parquet(out_dir / "hosts.parquet", index=False)
    if not keep_tmp:
        shutil.rmtree(tmp_dir)

    report = {
        "source": str(csv_path),
        "block_ms": BLOCK_MS,
        **totals,
        "n_blocks": len(blocks_seen),
        "n_hosts": len(hosts),
        "labels": label_counts,
        "attack_groups": group_counts,
        "n_features": len(FEATURE_COLUMNS),
        "features": list(FEATURE_COLUMNS),
        "dropped_columns": list(DROPPED_COLUMNS),
    }
    (out_dir / "prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def iter_blocks(
    dataset_dir: str | Path, columns: list[str] | None = None
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (block_id, frame) in time order from a prepared dataset directory."""
    for path in sorted((Path(dataset_dir) / "flows").glob("block-*.parquet")):
        block = int(path.stem.split("-")[1])
        yield block, pd.read_parquet(path, columns=columns)

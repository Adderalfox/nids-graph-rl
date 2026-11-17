"""Schema and quality audit for NetFlow-style NIDS datasets (Phase 1a).

Streams a (possibly multi-GB) CSV in chunks and reports everything the later
phases depend on: which identity/time columns exist, label balance, missing and
infinite values, constant columns, host counts, whether rows are time-ordered,
and how many flows fall into fixed time windows.

Nothing here modifies data. The report is plain JSON so it can be committed to
docs and pasted into a chat or paper appendix.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_IP = "IPV4_SRC_ADDR"
DST_IP = "IPV4_DST_ADDR"
SRC_PORT = "L4_SRC_PORT"
DST_PORT = "L4_DST_PORT"
FLOW_START = "FLOW_START_MILLISECONDS"
FLOW_END = "FLOW_END_MILLISECONDS"
LABEL = "Label"
ATTACK = "Attack"


def _find(columns: list[str], name: str) -> str | None:
    """Case-insensitive column lookup; returns the actual column name or None."""
    lookup = {c.lower(): c for c in columns}
    return lookup.get(name.lower())


def _num(value: Any) -> float | None:
    """JSON-safe float (NaN/inf become None)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def audit_csv(
    path: str | Path,
    *,
    chunksize: int = 500_000,
    window_seconds: tuple[int, ...] = (10, 60),
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Audit one dataset CSV (plain, .gz or single-file .zip) and return a report dict."""
    path = Path(path)

    rows = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    nan_counts: Counter[str] = Counter()
    inf_counts: Counter[str] = Counter()
    col_min: dict[str, float] = {}
    col_max: dict[str, float] = {}
    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    src_hosts: set[str] = set()
    dst_hosts: set[str] = set()

    t_min = np.inf
    t_max = -np.inf
    backward_steps = 0
    prev_last: float | None = None
    window_counts: dict[int, Counter[int]] = {w: Counter() for w in window_seconds}

    names: dict[str, str | None] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if max_rows is not None:
            remaining = max_rows - rows
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]

        if not columns:
            columns = list(chunk.columns)
            dtypes = {c: str(t) for c, t in chunk.dtypes.items()}
            names = {
                key: _find(columns, key)
                for key in (SRC_IP, DST_IP, SRC_PORT, DST_PORT, FLOW_START, FLOW_END, LABEL, ATTACK)
            }

        rows += len(chunk)

        for col, n in chunk.isna().sum().items():
            if n:
                nan_counts[str(col)] += int(n)

        numeric = chunk.select_dtypes(include="number")
        if not numeric.empty:
            values = numeric.to_numpy(dtype=float)
            inf_mask = np.isinf(values)
            for col, n in zip(numeric.columns, inf_mask.sum(axis=0), strict=True):
                if n:
                    inf_counts[str(col)] += int(n)
            finite = numeric.replace([np.inf, -np.inf], np.nan)
            for col, lo, hi in zip(numeric.columns, finite.min(), finite.max(), strict=True):
                col = str(col)
                if np.isfinite(lo):
                    col_min[col] = min(col_min.get(col, np.inf), float(lo))
                if np.isfinite(hi):
                    col_max[col] = max(col_max.get(col, -np.inf), float(hi))

        if names[LABEL]:
            label_counts.update(
                {str(k): int(v) for k, v in chunk[names[LABEL]].value_counts().items()}
            )
        if names[ATTACK]:
            attack_counts.update(
                {str(k): int(v) for k, v in chunk[names[ATTACK]].value_counts().items()}
            )
        if names[SRC_IP]:
            src_hosts.update(chunk[names[SRC_IP]].dropna().astype(str).unique())
        if names[DST_IP]:
            dst_hosts.update(chunk[names[DST_IP]].dropna().astype(str).unique())

        if names[FLOW_START]:
            start = pd.to_numeric(chunk[names[FLOW_START]], errors="coerce").to_numpy(dtype=float)
            start = start[np.isfinite(start)]
            if start.size:
                t_min = min(t_min, float(start.min()))
                t_max = max(t_max, float(start.max()))
                backward_steps += int(np.sum(np.diff(start) < 0))
                if prev_last is not None and start[0] < prev_last:
                    backward_steps += 1
                prev_last = float(start[-1])
                for w in window_seconds:
                    ids = (start // (w * 1000)).astype(np.int64)
                    uniq, counts = np.unique(ids, return_counts=True)
                    for u, c in zip(uniq.tolist(), counts.tolist(), strict=True):
                        window_counts[w][u] += c

    has_time = bool(names.get(FLOW_START) and np.isfinite(t_min))

    attack_rate = None
    if label_counts and rows:
        attack_rate = _num(sum(v for k, v in label_counts.items() if k not in {"0", "0.0"}) / rows)

    column_report = {
        c: {
            "dtype": dtypes[c],
            "nan": int(nan_counts.get(c, 0)),
            "inf": int(inf_counts.get(c, 0)),
            "min": _num(col_min.get(c)),
            "max": _num(col_max.get(c)),
        }
        for c in columns
    }
    constant = [c for c in columns if c in col_min and col_min[c] == col_max.get(c)]

    time_report: dict[str, Any] = {"present": has_time}
    windows_report: dict[str, Any] = {}
    if has_time:
        time_report.update(
            start_min_ms=_num(t_min),
            start_max_ms=_num(t_max),
            span_hours=_num((t_max - t_min) / 3_600_000),
            backward_steps=backward_steps,
            sorted_by_start=backward_steps == 0,
        )
        for w, counter in window_counts.items():
            sizes = np.array(list(counter.values()), dtype=float)
            windows_report[str(w)] = {
                "n_windows": int(sizes.size),
                "flows_mean": _num(sizes.mean()),
                "flows_median": _num(np.median(sizes)),
                "flows_p95": _num(np.percentile(sizes, 95)),
                "flows_max": _num(sizes.max()),
            }

    return {
        "file": str(path),
        "rows": rows,
        "n_columns": len(columns),
        "has_ip": bool(names.get(SRC_IP) and names.get(DST_IP)),
        "has_ports": bool(names.get(SRC_PORT) and names.get(DST_PORT)),
        "has_timestamps": has_time,
        "has_label": bool(names.get(LABEL)),
        "has_attack_category": bool(names.get(ATTACK)),
        "labels": dict(label_counts),
        "attack_categories": dict(attack_counts),
        "attack_rate": attack_rate,
        "hosts": {
            "unique_src": len(src_hosts),
            "unique_dst": len(dst_hosts),
            "unique_total": len(src_hosts | dst_hosts),
        },
        "time": time_report,
        "windows": windows_report,
        "constant_columns": constant,
        "columns_with_nan": sorted(nan_counts),
        "columns_with_inf": sorted(inf_counts),
        "columns": column_report,
    }


def format_report(report: dict[str, Any]) -> str:
    """Short human-readable summary of an audit report."""
    lines = [
        f"file            : {report['file']}",
        f"rows / columns  : {report['rows']:,} / {report['n_columns']}",
        f"has IPs / ports : {report['has_ip']} / {report['has_ports']}",
        f"has timestamps  : {report['has_timestamps']}",
        f"label / attack  : {report['has_label']} / {report['has_attack_category']}",
        f"attack rate     : {report['attack_rate']}",
        f"labels          : {report['labels']}",
        f"hosts (unique)  : {report['hosts']['unique_total']:,}",
    ]
    t = report["time"]
    if t.get("present"):
        lines.append(
            f"time span (h)   : {t['span_hours']:.2f}  sorted_by_start={t['sorted_by_start']}"
        )
        for w, s in report["windows"].items():
            lines.append(
                f"window {w:>3}s     : {s['n_windows']:,} windows, "
                f"median {s['flows_median']:.0f}, p95 {s['flows_p95']:.0f}, "
                f"max {s['flows_max']:.0f} flows"
            )
    lines.append(f"constant cols   : {report['constant_columns']}")
    lines.append(f"cols with NaN   : {report['columns_with_nan']}")
    lines.append(f"cols with inf   : {report['columns_with_inf']}")
    return "\n".join(lines)

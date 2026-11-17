import numpy as np
import pandas as pd

from nidsrl.data.audit import audit_csv


def _make_df(n: int = 20, shuffled: bool = False, with_time: bool = True) -> pd.DataFrame:
    start = np.arange(n) * 4000 + 1_600_000_000_000  # one flow every 4 s
    if shuffled:
        start = start[::-1]
    df = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": [f"10.0.0.{i % 3}" for i in range(n)],
            "L4_SRC_PORT": np.arange(n) + 1000,
            "IPV4_DST_ADDR": [f"10.0.1.{i % 2}" for i in range(n)],
            "L4_DST_PORT": 80,
            "IN_BYTES": np.arange(n, dtype=float),
            "CONST": 1,
            "Label": [0] * (n - 5) + [1] * 5,
            "Attack": ["Benign"] * (n - 5) + ["DoS"] * 5,
        }
    )
    if with_time:
        df["FLOW_START_MILLISECONDS"] = start
        df["FLOW_END_MILLISECONDS"] = start + 100
    return df


def test_basic_report_across_chunk_boundaries(tmp_path):
    f = tmp_path / "d.csv"
    _make_df().to_csv(f, index=False)
    r = audit_csv(f, chunksize=7, window_seconds=(10,))
    assert r["rows"] == 20
    assert r["has_ip"] and r["has_ports"] and r["has_timestamps"]
    assert r["labels"] == {"0": 15, "1": 5}
    assert r["attack_rate"] == 0.25
    assert r["hosts"]["unique_src"] == 3 and r["hosts"]["unique_dst"] == 2
    assert r["time"]["sorted_by_start"] is True
    assert "CONST" in r["constant_columns"]
    # 20 flows, 4 s apart, 10 s windows -> flows spread over 8 windows of 2-3 flows
    assert r["windows"]["10"]["flows_max"] <= 3


def test_unsorted_time_is_detected(tmp_path):
    f = tmp_path / "d.csv"
    _make_df(shuffled=True).to_csv(f, index=False)
    r = audit_csv(f, chunksize=7, window_seconds=(10,))
    assert r["time"]["sorted_by_start"] is False
    assert r["time"]["backward_steps"] == 19


def test_missing_timestamps_reported(tmp_path):
    f = tmp_path / "d.csv"
    _make_df(with_time=False).to_csv(f, index=False)
    r = audit_csv(f)
    assert r["has_timestamps"] is False
    assert r["windows"] == {}


def test_nan_and_inf_are_counted(tmp_path):
    df = _make_df()
    df.loc[2, "IN_BYTES"] = np.inf
    df.loc[3, "IN_BYTES"] = np.nan
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    r = audit_csv(f, chunksize=6)
    assert r["columns"]["IN_BYTES"]["inf"] == 1
    assert r["columns"]["IN_BYTES"]["nan"] == 1
    assert "IN_BYTES" in r["columns_with_inf"]


def test_max_rows_limits_audit(tmp_path):
    f = tmp_path / "d.csv"
    _make_df().to_csv(f, index=False)
    r = audit_csv(f, chunksize=7, max_rows=10)
    assert r["rows"] == 10

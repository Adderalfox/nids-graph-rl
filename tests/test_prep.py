import numpy as np
import pandas as pd
import pytest

from nidsrl.data.prep import (
    BLOCK_MS,
    DROPPED_COLUMNS,
    FEATURE_COLUMNS,
    LOG1P_COLUMNS,
    META_COLUMNS,
    SCALED_COLUMNS,
    HostIndex,
    iter_blocks,
    prepare_dataset,
    transform_chunk,
)
from nidsrl.data.taxonomy import coarse_attack, is_known

BASE = 1_526_344_000_000


def make_raw(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # three different hour blocks, deliberately shuffled (unsorted export)
    start = BASE + rng.integers(0, 3, n) * BLOCK_MS + rng.integers(0, BLOCK_MS, n)
    rng.shuffle(start)
    data: dict[str, object] = {
        "FLOW_START_MILLISECONDS": start,
        "FLOW_END_MILLISECONDS": start + 50,
        "IPV4_SRC_ADDR": [f"10.0.0.{i % 4}" for i in range(n)],
        "L4_SRC_PORT": rng.integers(0, 65535, n),
        "IPV4_DST_ADDR": [f"192.168.1.{i % 3}" for i in range(n)],
        "L4_DST_PORT": rng.integers(0, 1024, n),
        "PROTOCOL": rng.choice([1, 6, 17, 47], n),
        "Label": (np.arange(n) % 5 == 0).astype(int),
        "Attack": ["DDoS_attacks-LOIC-HTTP" if i % 5 == 0 else "Benign" for i in range(n)],
        "CLIENT_TCP_FLAGS": np.full(n, 18),  # SYN + ACK
        "SERVER_TCP_FLAGS": np.zeros(n, dtype=int),
    }
    for col in LOG1P_COLUMNS:
        data[col] = rng.integers(0, 5000, n)
    for col in SCALED_COLUMNS:
        data[col] = rng.integers(0, 255, n)
    # dropped columns, including the problematic inf/NaN ones
    for col in DROPPED_COLUMNS:
        data[col] = rng.random(n)
    data["SRC_TO_DST_SECOND_BYTES"] = np.where(np.arange(n) % 7 == 0, np.inf, 1.0)
    data["DST_TO_SRC_SECOND_BYTES"] = np.where(np.arange(n) % 11 == 0, np.nan, 1.0)
    return pd.DataFrame(data)


def test_feature_set_is_identity_free_and_dropped_columns_absent():
    banned_tokens = {"addr", "port", "start", "end", "timestamp", "dns", "l7"}
    for name in FEATURE_COLUMNS:
        assert not (set(name.split("_")) & banned_tokens), name
        assert "second_bytes" not in name
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 55


def test_transform_decodes_flags_and_logs():
    raw = make_raw(10)
    frame, counters = transform_chunk(raw, HostIndex())
    # 18 = 0b10010 -> syn (bit 1) and ack (bit 4)
    assert (frame["client_syn"] == 1).all() and (frame["client_ack"] == 1).all()
    assert (frame["client_fin"] == 0).all() and (frame["server_syn"] == 0).all()
    assert np.allclose(frame["log_in_bytes"], np.log1p(raw["IN_BYTES"]), atol=1e-5)
    assert (
        frame["proto_tcp"]
        .add(frame["proto_udp"])
        .add(frame["proto_icmp"])
        .add(frame["proto_other"])
        .eq(1)
        .all()
    )
    assert counters["dropped_bad_time"] == 0
    assert list(frame.columns) == list(META_COLUMNS) + list(FEATURE_COLUMNS)


def test_non_finite_values_are_neutralised_and_counted():
    raw = make_raw(10)
    raw["IN_BYTES"] = raw["IN_BYTES"].astype(float)
    raw["TCP_WIN_MAX_IN"] = raw["TCP_WIN_MAX_IN"].astype(float)
    raw.loc[0, "IN_BYTES"] = np.inf
    raw.loc[1, "TCP_WIN_MAX_IN"] = np.nan
    frame, counters = transform_chunk(raw, HostIndex())
    assert counters["nonfinite_replaced"] == 2
    assert np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all()


def test_bad_timestamps_are_dropped():
    raw = make_raw(10)
    raw.loc[3, "FLOW_START_MILLISECONDS"] = 0
    frame, counters = transform_chunk(raw, HostIndex())
    assert len(frame) == 9 and counters["dropped_bad_time"] == 1


def test_missing_required_column_raises():
    raw = make_raw(5).drop(columns=["IN_BYTES"])
    with pytest.raises(KeyError):
        transform_chunk(raw, HostIndex())


def test_host_ids_are_stable_across_chunks():
    h = HostIndex()
    a = h.encode(pd.Series(["a", "b", "a"]))
    b = h.encode(pd.Series(["b", "c", "a"]))
    assert list(a) == [0, 1, 0]
    assert list(b) == [1, 2, 0]
    assert len(h) == 3


def test_full_pipeline_sorts_blocks_and_preserves_rows(tmp_path):
    raw = make_raw(60)
    csv = tmp_path / "raw.csv"
    raw.to_csv(csv, index=False)
    out = tmp_path / "unsw"
    report = prepare_dataset(csv, out, chunksize=17)

    assert report["rows_in"] == report["rows_out"] == 60
    assert report["n_features"] == 55
    assert report["labels"] == {"0": 48, "1": 12}
    assert report["attack_groups"] == {"benign": 48, "dos": 12}
    assert not (out / "_tmp").exists()

    total, last_block = 0, -1
    for block, df in iter_blocks(out):
        assert block > last_block
        last_block = block
        assert df["start_ms"].is_monotonic_increasing
        assert (df["start_ms"] // BLOCK_MS == block).all()
        total += len(df)
    assert total == 60

    hosts = pd.read_parquet(out / "hosts.parquet")
    assert len(hosts) == report["n_hosts"] == 7  # 4 sources + 3 destinations


def test_existing_output_requires_overwrite(tmp_path):
    csv = tmp_path / "raw.csv"
    make_raw(10).to_csv(csv, index=False)
    out = tmp_path / "x"
    prepare_dataset(csv, out)
    with pytest.raises(FileExistsError):
        prepare_dataset(csv, out)
    prepare_dataset(csv, out, overwrite=True)


def test_taxonomy_covers_all_audited_attack_names():
    names = [
        "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Backdoor", "DoS", "Shellcode",
        "Analysis", "Worms", "scanning", "dos", "injection", "ddos", "password", "xss",
        "ransomware", "mitm", "FTP-BruteForce", "SSH-Bruteforce", "DoS_attacks-GoldenEye",
        "DoS_attacks-Slowloris", "DoS_attacks-SlowHTTPTest", "DoS_attacks-Hulk",
        "DDoS_attacks-LOIC-HTTP", "DDOS_attack-LOIC-UDP", "DDOS_attack-HOIC",
        "Brute_Force_-Web", "Brute_Force_-XSS", "SQL_Injection", "Infilteration", "Bot",
        "Theft", "Benign",
    ]  # fmt: skip
    assert all(is_known(n) for n in names)
    assert coarse_attack("DDOS_attack-HOIC") == "dos"
    assert coarse_attack("Brute_Force_-XSS") == "web"
    assert coarse_attack("never-seen") == "other"

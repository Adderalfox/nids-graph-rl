import numpy as np
import pandas as pd
import pytest

from nidsrl.data.hostaudit import (
    HOST,
    PAIR,
    SERVICE,
    audit_entities,
    audit_hosts,
    format_host_report,
)

W = 10_000  # 10 s windows
BASE = 420_000 * 3_600_000


def write_block(root, rows):
    d = root / "flows"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["window", "src_id", "label", "count"])
    df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "start_ms": BASE + df["window"].to_numpy() * W + 1,
            "src_id": df["src_id"].astype(np.int32),
            "label": df["label"].astype(np.uint8),
        }
    )
    out.to_parquet(d / "block-000420000.parquet", index=False)


@pytest.fixture
def audit(tmp_path):
    rows = []
    for w in range(5):  # host 1: attacker present in windows 0..4, also benign flows
        rows += [(w, 1, 1, 10), (w, 1, 0, 5)]
    for w in range(5):  # host 2: benign only
        rows += [(w, 2, 0, 20)]
    rows += [(2, 3, 1, 3)]  # host 3: one-shot attacker
    write_block(tmp_path, rows)
    return audit_hosts(tmp_path, window_s=10)


def test_counts_and_active_hosts(audit):
    assert audit["flows"] == 5 * 15 + 5 * 20 + 3
    assert audit["host_windows"] == 11
    # windows 0,1,3,4 have hosts {1,2}; window 2 has {1,2,3}
    assert audit["active_src_hosts_per_window"]["max"] == 3
    assert audit["active_src_hosts_per_window"]["median"] == 2


def test_purity(audit):
    p = audit["host_window_purity"]
    assert p["pure_benign_share"] == pytest.approx(5 / 11)  # host 2
    assert p["pure_attack_share"] == pytest.approx(1 / 11)  # host 3
    assert p["mixed_share"] == pytest.approx(5 / 11)  # host 1
    assert p["mixed_attack_fraction"]["median"] == pytest.approx(10 / 15)


def test_persistence_and_preventable_share(audit):
    ap = audit["attacker_persistence"]
    assert ap["attacker_host_blocks"] == 2
    assert ap["one_shot_share"] == pytest.approx(0.5)  # host 3
    # attack flows: host1 5*10=50, host3 3. After first window: host1 4*10=40, host3 0
    assert ap["preventable_attack_flow_share"] == pytest.approx(40 / 53)
    assert audit["attacker_hosts_per_attack_window"]["max"] == 2


def test_collateral(audit):
    # benign flows from host 1 after its first attack window: windows 1..4 -> 4*5 = 20
    c = audit["collateral"]
    assert c["benign_flows_from_attackers_after_first_window"] == 20
    assert c["share_of_all_benign_flows"] == pytest.approx(20 / (25 + 100))


def test_report_formats(audit):
    text = format_host_report("demo", audit)
    assert "PREVENTABLE" in text and "COLLATERAL" in text


def test_benign_only_dataset_has_no_attack_statistics(tmp_path):
    write_block(tmp_path, [(0, 1, 0, 4), (1, 2, 0, 4)])
    r = audit_hosts(tmp_path)
    assert r["attacker_persistence"]["preventable_attack_flow_share"] is None
    assert r["attacker_persistence"]["one_shot_share"] is None


def test_pair_granularity_separates_a_mixed_host(tmp_path):
    """Host 1 attacks dst 10 but serves benign traffic to dst 20: mixed per host, pure per pair."""
    d = tmp_path / "flows"
    d.mkdir()
    rows = []
    for w in range(4):
        rows += [(w, 1, 10, 80, 1, 10), (w, 1, 20, 443, 0, 8), (w, 2, 20, 443, 0, 5)]
    df = pd.DataFrame(rows, columns=["window", "src_id", "dst_id", "dst_port", "label", "count"])
    df = df.loc[df.index.repeat(df["count"])].reset_index(drop=True)
    pd.DataFrame(
        {
            "start_ms": BASE + df["window"].to_numpy() * W + 1,
            "src_id": df["src_id"].astype(np.int32),
            "dst_id": df["dst_id"].astype(np.int32),
            "dst_port": df["dst_port"].astype(np.uint16),
            "label": df["label"].astype(np.uint8),
        }
    ).to_parquet(d / "block-000420000.parquet", index=False)

    host = audit_entities(tmp_path, 10, HOST)
    pair = audit_entities(tmp_path, 10, PAIR)
    service = audit_entities(tmp_path, 10, SERVICE)
    assert host["host_window_purity"]["mixed_share"] == pytest.approx(4 / 8)
    assert pair["host_window_purity"]["mixed_share"] == 0.0
    assert service["host_window_purity"]["mixed_share"] == 0.0
    assert host["collateral"]["share_of_all_benign_flows"] > 0.4
    assert pair["collateral"]["share_of_all_benign_flows"] == 0.0
    assert pair["entity"] == "src_id+dst_id"
    assert pair["attacker_persistence"]["preventable_attack_flow_share"] == pytest.approx(0.75)


def test_timed_block_oracle_simulation(audit):
    tb = audit["timed_block"]
    # host 1 attacks in windows 0..4 (10 attack + 5 benign each); host 3 one-shot at w2.
    # k=1: blocks at w0, w2, w4 prevent windows 1 and 3 -> 20 attack flows, 10 benign lost.
    assert tb["1"]["prevented_attack_share"] == pytest.approx(20 / 53)
    assert tb["1"]["collateral_benign_share"] == pytest.approx(10 / 125)
    assert tb["1"]["blocks"] == 4  # three for host 1, one for host 3
    # k=3: block at w0 covers w1..w3 -> 30 attack flows, 15 benign lost; w4 re-triggers
    assert tb["3"]["prevented_attack_share"] == pytest.approx(30 / 53)
    assert tb["3"]["collateral_benign_share"] == pytest.approx(15 / 125)
    assert tb["3"]["blocks"] == 3
    assert tb["1"]["seconds"] == 10


def test_longer_blocks_prevent_more_but_cost_more(audit):
    tb = audit["timed_block"]
    assert (
        tb["30"]["prevented_attack_share"]
        >= tb["6"]["prevented_attack_share"]
        >= tb["3"]["prevented_attack_share"]
        >= tb["1"]["prevented_attack_share"]
    )
    assert (
        tb["30"]["collateral_benign_share"]
        >= tb["6"]["collateral_benign_share"]
        >= tb["3"]["collateral_benign_share"]
        >= tb["1"]["collateral_benign_share"]
    )


def test_flow_level_ceiling_with_mixed_window(tmp_path):
    # window 0: host 1 has 2 attack + 8 benign flows (majority benign -> attack flows missed)
    # window 1: host 1 has 9 attack + 1 benign (majority attack -> that benign flow flagged)
    # window 2: host 2 benign only (20)
    write_block(tmp_path, [(0, 1, 1, 2), (0, 1, 0, 8), (1, 1, 1, 9), (1, 1, 0, 1), (2, 2, 0, 20)])
    r = audit_hosts(tmp_path)
    maj = r["flow_level_ceiling"]["majority_rule"]
    # tp=9, fn=2, fp=1, tn=28 -> tpr 9/11, fpr 1/29
    assert maj["tpr"] == pytest.approx(9 / 11)
    assert maj["fpr"] == pytest.approx(1 / 29)
    anyr = r["flow_level_ceiling"]["any_attack_rule"]
    # every window with an attack is flagged: all 11 attack flows caught, 8 + 1 benign flagged
    assert anyr["tpr"] == pytest.approx(1.0)
    assert anyr["fpr"] == pytest.approx(9 / 29)
    assert "CEILING" in format_host_report("demo", r)

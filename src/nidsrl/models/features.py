"""A-priori feature groups for ablations and shortcut diagnosis.

Groups are defined from domain knowledge BEFORE looking at any transfer result:
`host_fingerprint` (TTL and TCP window sizes) mostly reflects the operating system
and network path of an endpoint, i.e. the testbed, not the behaviour of an attack.
Ablations are reported in full; the feature policy of the main model must not be
chosen by looking at target-dataset performance (see docs/evaluation.md).
"""

from __future__ import annotations

from nidsrl.data.prep import FEATURE_COLUMNS

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "volume": (
        "log_in_bytes",
        "log_in_pkts",
        "log_out_bytes",
        "log_out_pkts",
        "log_src_to_dst_avg_throughput",
        "log_dst_to_src_avg_throughput",
    ),
    "timing": (
        "log_flow_duration_milliseconds",
        "log_duration_in",
        "log_duration_out",
        "log_src_to_dst_iat_min",
        "log_src_to_dst_iat_max",
        "log_src_to_dst_iat_avg",
        "log_src_to_dst_iat_stddev",
        "log_dst_to_src_iat_min",
        "log_dst_to_src_iat_max",
        "log_dst_to_src_iat_avg",
        "log_dst_to_src_iat_stddev",
    ),
    "pkt_size": (
        "log_longest_flow_pkt",
        "log_shortest_flow_pkt",
        "log_min_ip_pkt_len",
        "log_max_ip_pkt_len",
        "log_num_pkts_up_to_128_bytes",
        "log_num_pkts_128_to_256_bytes",
        "log_num_pkts_256_to_512_bytes",
        "log_num_pkts_512_to_1024_bytes",
        "log_num_pkts_1024_to_1514_bytes",
    ),
    "retrans": (
        "log_retransmitted_in_bytes",
        "log_retransmitted_in_pkts",
        "log_retransmitted_out_bytes",
        "log_retransmitted_out_pkts",
    ),
    "host_fingerprint": ("min_ttl", "max_ttl", "tcp_win_max_in", "tcp_win_max_out"),
    "tcp_flags": tuple(
        f"{side}_{b}"
        for side in ("client", "server")
        for b in ("fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr")
    ),
    "proto": ("proto_tcp", "proto_udp", "proto_icmp", "proto_other", "icmp_ipv4_type"),
}


def kept_feature_indices(
    drop_groups: tuple[str, ...] | list[str] = (),
    keep_groups: tuple[str, ...] | list[str] = (),
) -> list[int]:
    """Column indices of FEATURE_COLUMNS that remain.

    keep_groups (if given) keeps ONLY those groups; drop_groups then removes groups.
    """
    unknown = (set(drop_groups) | set(keep_groups)) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(
            f"Unknown feature groups {sorted(unknown)}; choose from {sorted(FEATURE_GROUPS)}"
        )
    dropped = {name for g in drop_groups for name in FEATURE_GROUPS[g]}
    allowed = (
        {name for g in keep_groups for name in FEATURE_GROUPS[g]}
        if keep_groups
        else set(FEATURE_COLUMNS)
    )
    return [i for i, name in enumerate(FEATURE_COLUMNS) if name in allowed and name not in dropped]

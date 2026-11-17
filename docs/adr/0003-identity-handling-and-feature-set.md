# ADR 0003: Host identity handling and the flow feature set

Status: accepted

## Context
Windows are graph snapshots with hosts as nodes. Two things pull in opposite
directions: temporal learning needs to know that a node in window t is the same
host as a node in window t+1, while cross-dataset generalization forbids the
model from learning host identities (IP addresses).

## Decision
1. Each dataset gets a stable integer host id (first-appearance order), stored
   as `src_id` / `dst_id` plus a `hosts.parquet` id-to-IP table (for
   visualization only).
2. Host ids are used only as keys: to look up per-host temporal state and to
   compute identity-free history features (e.g. change in degree, new-neighbour
   ratio versus the previous window). They are never model inputs and never index
   an embedding table.
3. Within a window, nodes are re-indexed contiguously for computation, with an
   explicit local-to-stable mapping so state carries across windows. Hosts that
   appear or disappear are handled explicitly (Phase 3).
4. Temporal state is reset at the start of every dataset. Host ids are
   dataset-local and meaningless across datasets.
5. Flow features exclude IPs, ports and absolute time. Dropped columns:
   `*_SECOND_BYTES` (inconsistent across datasets), `L7_PROTO`, `TCP_FLAGS`,
   `ICMP_TYPE`, all `DNS_*`, `FTP_COMMAND_RET_CODE`. Heavy-tailed columns are
   log1p-transformed, bounded columns divided by their natural maximum, TCP
   flag bitmasks decoded to bits, protocol one-hot (tcp/udp/icmp/other).
   Standardization is deferred to Phase 1c and fit on training data only.
6. Ports remain available as metadata for a later ablation.

## Consequences
- Temporal modelling is possible without identity leakage.
- 55 features, identical across datasets.
- Feature choices rely only on dataset-independent facts (constant, inconsistent
  or ID-like columns), not on target-domain statistics.

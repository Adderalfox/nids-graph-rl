# ADR 0002: Use the NetFlow v3 (NF3) dataset family

Status: accepted (confirm in the Phase 1a audit)

## Context
ADR 0001 requires IP addresses (for graph structure) and flow timestamps (for
rolling windows). The NetFlow v2 datasets provide IPs and ports, but their 43
features and sample rows show no flow start/end time columns. The NetFlow v3
datasets (53 features) add FLOW_START_MILLISECONDS and FLOW_END_MILLISECONDS
plus inter-arrival-time features, and exist for the same four source datasets.

## Decision
Use NF3 versions: NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-CSE-CIC-IDS2018-v3,
NF-BoT-IoT-v3. Binary labels (benign/attack) are the primary target because
attack-category names differ across datasets.

## Consequences
- Real time windows instead of row-count windows.
- If the audit shows missing or unsorted timestamps in any dataset, that
  dataset is documented as a limitation or dropped from the transfer matrix.
- The NF3 feature set must be confirmed identical across the four datasets in
  the audit before cross-dataset training.

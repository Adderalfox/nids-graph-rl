# ADR 0001: Graph representation and time handling

Status: accepted (revisit after Phase 1 schema audit)

## Context
The goal is cross-dataset generalization. Representations tied to
dataset-specific identifiers (IP addresses, port numbers, absolute time) tend
to memorize a dataset rather than transfer.

## Decision
- Nodes are hosts; edges are flows carrying standardized flow features.
- Time is handled with fixed rolling windows and relative encodings
  (inter-arrival gaps, duration, position within window).
- IPs are used only to build structure and are never node features.

## Consequences
- Cheaper than continuous-time graphs on a 4 GB GPU and maps cleanly to RL steps.
- Window size becomes a hyperparameter to ablate (Phase 6).
- Requires datasets that retain IPs and timestamps (checked in Phase 1).

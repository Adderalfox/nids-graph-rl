# ADR 0005: Stratified purged splits; transfer evaluated on the whole target

Status: accepted (supersedes the chronological split in ADR 0004)

## Context
The first Phase 1c run used contiguous chronological splits. On the real data it
failed: attacks are scheduled in time, so CIC validation held only `malware` at
4% attacks and ToN validation held only `dos`, while ToN test held families that
never appeared in training. Models could not be tuned or fairly evaluated.

## Decision
1. The timeline is cut into 5-minute segments grouped into 30-minute runs.
2. Runs are assigned to train/val/test (70/10/20) by a deterministic greedy
   stratified assignment over benign flows and every attack group.
3. One segment (5 min) is purged wherever the split changes, so no train and
   test flow can share a 10 s or 60 s window, a long-lived flow (max about
   2 min) or temporal state.
4. Runs are the episode unit for the temporal model: sequences never cross a run
   boundary and temporal state resets at each run start.
5. Zero-shot transfer is evaluated on the WHOLE target dataset (nothing from the
   target is used for training or tuning). In-dataset reference results use the
   source's own test rows. BoT-IoT is always evaluated on all rows.

## Consequences
- Every split contains every attack group, so validation is meaningful.
- Caveat: the same attack campaign can appear in train and test (in runs that are
  purged apart), so in-dataset scores are an optimistic reference. The headline
  result is zero-shot transfer, which has no such overlap. State this in the paper.
- Roughly 10% of segments are purged.
- Row membership is computed from `start_ms` via `assign_rows`; plans store run
  ids and purged segment ids, not per-row labels.

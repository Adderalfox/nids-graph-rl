# Roadmap and decision log

## Phases

| # | Phase | Exit criterion | Status |
|---|-------|----------------|--------|
| 0 | Foundations: repo, config, seeding, logging, CI | `pytest` and `ruff` pass; `check_env.py` OK | in progress |
| 1 | Data and evaluation protocol (NF-v2 datasets, leakage-free splits, metrics) | data report, split files, leakage tests | not started |
| 2 | Environment (true MDP) and baselines (RF/XGB/MLP, MLP+DQN) | baseline table incl. cross-dataset matrix | not started |
| 3 | Windowed host/flow graph builder and encoder | graph encoder vs MLP, transfer gap | not started |
| 4 | Dueling Double DQN on graph embeddings, PER baseline | B vs A over 5 seeds | not started |
| 5 | Topology-aware NSGA-II replay | C vs B, replay overhead | not started |
| 6 | Ablations, unseen families, latency, significance | full results and figures | not started |
| 7 | Paper and portfolio packaging | submission-ready draft, public repo | not started |

## Decisions so far

- Graph: hosts as nodes, flows as edges, rolling fixed-size time windows.
- No IP or port identity features; relative (not absolute) time encodings.
- Datasets: NetFlow-v2 family (UNSW-NB15, CSE-CIC-IDS2018, ToN-IoT, BoT-IoT).
  Train on one, test zero-shot on the others. Verify IP/timestamp availability
  in the Phase 1 schema audit.
- Stack: temporal graph encoder + Dueling Double DQN + topology-aware
  NSGA-II replay. Experiments: A (MLP+DQN+random), B (graph+D3QN+PER),
  C (graph+D3QN+topology-aware NSGA-II).
- Code is written from scratch; the earlier DQN prototype is not reused.

## Resume summary (paste into a new chat)

> Project: NIDS with graph-based deep RL aimed at cross-dataset generalization;
> research paper plus flagship masters-admission project (professional repo
> standards). Aron, Windows, RTX 3050 Ti 4 GB, PyTorch. Repo: nids-graph-rl
> (src layout, package `nidsrl`, YAML configs, pytest, ruff, CI).
> Decisions: hosts=nodes, flows=edges, rolling windows, relative time, no
> IP/port identity features; NF-v2 datasets, train one / zero-shot others;
> temporal graph encoder + Dueling Double DQN + topology-aware NSGA-II replay.
> Phases: 0 foundations, 1 data+eval protocol, 2 env+baselines, 3 graph encoder,
> 4 RL agent+PER, 5 topology-aware replay, 6 ablations, 7 paper.
> Current phase: Phase 0 (foundations) delivered as a zip. Resume from Phase ___.

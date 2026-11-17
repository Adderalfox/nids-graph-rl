# Roadmap and decision log

## Phases

| # | Phase | Exit criterion | Status |
|---|-------|----------------|--------|
| 0 | Foundations: repo, config, seeding, logging, CI | `pytest` and `ruff` pass; `check_env.py` OK | done |
| 1 | Data and evaluation protocol | data report, split files, leakage tests | in progress |
| 1a | Download NF3 datasets and run the schema audit | audit JSON for each dataset | done |
| 1b | Clean, sort, host ids, identity-free features (Parquet) | prepared datasets + tests | code delivered, awaiting runs |
| 1c | Time-blocked splits, window sampling, scaler, cross-dataset protocol | split files + leakage tests | not started |
| 1d | Metrics module and evaluation harness skeleton | metrics tests | not started |
| 2 | Environment (true MDP) and baselines (RF/XGB/MLP, MLP+DQN) | baseline table incl. cross-dataset matrix | not started |
| 3 | Windowed host/flow graph builder and encoder | graph encoder vs MLP, transfer gap | not started |
| 4 | Dueling Double DQN on graph embeddings, PER baseline | B vs A over 5 seeds | not started |
| 5 | Topology-aware NSGA-II replay | C vs B, replay overhead | not started |
| 6 | Ablations, unseen families, latency, significance | full results and figures | not started |
| 7 | Paper and portfolio packaging | submission-ready draft, public repo | not started |

## Decisions so far

- Graph: hosts as nodes, flows as edges, rolling 10 s windows (window size is
  a hyperparameter to ablate).
- Stable per-dataset host ids are keys for temporal state only, never features
  or embeddings (ADR 0003). Temporal state resets per dataset.
- No IP, port or absolute-time features; relative time encodings.
- Datasets: NetFlow v3 family (v2 lacks timestamps, ADR 0002). UNSW, CIC and
  ToN are sources and targets; BoT is target-only (99.7% attacks). Binary labels
  first; coarse attack groups for per-group recall.
- Compute budget: 16 GB RAM, 4 GB RTX 3050 Ti, about 47 GB free disk. Data is
  processed in streaming chunks; whole windows (never single flows) are sampled.
- Stack: temporal graph encoder + Dueling Double DQN + topology-aware
  NSGA-II replay. Experiments: A (MLP+DQN+random), B (graph+D3QN+PER),
  C (graph+D3QN+topology-aware NSGA-II).
- Tooling: uv. Code written from scratch; the old DQN prototype is not reused.

## Resume summary (paste into a new chat)

> Project: NIDS with graph-based deep RL aimed at cross-dataset generalization;
> research paper plus flagship masters-admission project (professional repo
> standards). Aron, Windows + Git Bash, 16 GB RAM, RTX 3050 Ti 4 GB, ~47 GB free
> disk, PyTorch, uv. Repo: nids-graph-rl (src layout, package `nidsrl`, YAML
> configs, pytest, ruff, CI).
> Decisions: hosts=nodes, flows=edges, 10 s windows; stable per-dataset host ids
> used only as state keys (never features); no IP/port/absolute-time features;
> NetFlow v3 datasets (identical 55-column schema), UNSW/CIC/ToN sources and
> targets, BoT target-only; 55 identity-free features (log1p, bit-decoded flags,
> protocol one-hot) stored as time-block Parquet; temporal graph encoder +
> Dueling Double DQN + topology-aware NSGA-II replay.
> Phases: 0 done, 1a audit done, 1b prep code delivered (user must run it on all
> four datasets), 1c splits+scaler+protocol, 1d metrics, 2 env+baselines,
> 3 graph encoder, 4 RL agent+PER, 5 topology-aware replay, 6 ablations, 7 paper.
> Current phase: 1b (awaiting prepare_data runs and their prep_report.json files).
> Resume from Phase ___.

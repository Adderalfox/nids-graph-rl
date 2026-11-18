# Roadmap and decision log

## Phases

| # | Phase | Exit criterion | Status |
|---|-------|----------------|--------|
| 0 | Foundations: repo, config, seeding, logging, CI | `pytest` and `ruff` pass; `check_env.py` OK | done |
| 1 | Data and evaluation protocol | data report, split files, leakage tests | in progress |
| 1a | Download NF3 datasets and run the schema audit | audit JSON for each dataset | done |
| 1b | Clean, sort, host ids, identity-free features (Parquet) | prepared datasets + tests | done |
| 1c | Stratified purged splits, train-only scaler, cross-dataset protocol | split files + leakage tests | done |
| 1d | Metrics module, split loader and evaluation harness | metrics + harness tests | done |
| 2a | Supervised baselines (logreg, RF, HistGB) + results matrix + feature-group ablations | baseline transfer table | first run done (ton/hgb); full runs + ablations pending |
| 2b-0 | Entity-window audit: decision granularity, block semantics, flow-level ceiling | audit numbers incl. ceiling | host/pair/service + timed blocks done; ceiling pending |
| 2b | Host-window environment (true MDP) and MLP+DQN baseline | experiment A numbers | blocked on 2b-0 results |
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
- Protocol (ADR 0004/0005): stratified purged splits (30-min runs, 5-min purge;
  chronological splits failed because attack campaigns are scheduled in time);
  hyperparameters on source validation only; zero-shot transfer evaluated on the
  whole target dataset, in-dataset reference on the source test rows; scaler fit
  on source train only; per-group recall on shared groups, separate detection
  rate for unseen groups. Window sampling moves to Phase 3.
- Metrics (docs/evaluation.md): no accuracy. TPR, FPR, macro-F1, ROC-AUC, TPR at
  FPR 1% and 0.1%, PR-AUC with prevalence; per-group recall (shared) and detection
  rate (unseen) for groups with at least 500 flows; inference throughput. Models
  plug into the harness as predict(X) -> (y_pred, y_score).
- Baselines (docs/baselines.md): fit on a category-capped train sample (150k per
  category), decision threshold chosen on a source-val sample at 1% FPR, evaluated
  on every protocol pair; 5 seeds for reported numbers.
- Host audit finding: preventable attack share ~99% on all four datasets, but collateral
  of blocking attacker hosts is 0% (unsw), 1.8% (cic), 54% (ton), 86% (bot) because
  attacker hosts also send benign traffic, and 62.7% of CIC attacker hosts are one-shot.
  Host-level reward would differ by dataset and confound transfer. Proposal: decide per
  host PAIR (src->dst) per window; verify with `audit_hosts.py --granularity pair|service`.
- Ablations (ton, hgb; docs/findings.md): host_fingerprint (TTL, TCP window) is a testbed
  shortcut (alone: unsw AUC 0.10, bot 0.86; dropping it: unsw 0.53, cic 0.80, in-dataset
  0.96). Behavioural groups reach ~0.78 AUC on cic but transfer to unsw is weak, and TPR at
  1% FPR stays below 0.02 on cic. Thresholds do not transfer. BoT likely shares a lab with
  ToN (hypothesis), so it is not an independent unseen domain.
- Audit: no single granularity works (collateral of permanent blocks: host 0/1.8/54/86%,
  pair 0/0.4/49.6/68.8%, service 0/0.2/12.5/18.3% for unsw/cic/ton/bot; service entities
  explode to 100k+ per window). Direction: host-level decisions with EXPIRING blocks and
  class-normalised rewards; confirm with the timed-block audit.
- Decision: the RL agent acts per host per window (monitor / inspect / block),
  a simulated response environment (docs/host_window_mdp.md). Action semantics and
  reward parameters are finalized only after the 2b-0 audit.
- Finding (ton/hgb, stride 10): in-dataset AUC 0.93 but transfer AUC 0.16 (unsw),
  0.38 (cic), 0.43 (bot), i.e. inverted rankings: the model learned dataset-specific
  shortcuts. Hypothesis: host_fingerprint features (TTL, TCP window). Test by
  feature-group ablation; do not tune the feature policy on target results.
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
> Prepared datasets exist for all four (66.9M flows; attack-group overlap is
> partial: exploit only in unsw, bruteforce only cic/ton, web only cic/ton).
> Split results (real data): every source split has every attack group; ~65/9/18
> train/val/test, ~7% purged; CIC web has only 89 flows in val; UNSW is data-poor
> for dos/malware (expect weakest transfer from unsw; ton->cic best).
> Baseline finding: HGB trained on ton has in-dataset AUC 0.93 but inverted transfer
> AUC (0.16 unsw, 0.38 cic, 0.43 bot): shortcut learning; ablating feature groups next.
> Findings (docs/findings.md): host_fingerprint features are a testbed shortcut; flow-level
> baselines transfer poorly even without them; audit says permanent host/pair blocks have
> large collateral on ton/bot, so blocks must expire; rewards class-normalised.
> Phases: 0, 1 (a-d) done; 2a supervised baselines + ablations (in progress),
> 2b-0 host-window audit (code delivered), 2b host-window env + MLP+DQN,
> 3 graph encoder, 4 RL agent+PER, 5 topology-aware replay, 6 ablations, 7 paper.
> Provisional MDP: per host pair per 10 s window; monitor / inspect (10 s quarantine) /
> block (60 s); class-normalised rewards; flow-level evaluation with alert = not monitor.
> Timed-block audit done (docs/findings.md); flow-level ceiling audit pending.
> Current phase: 2b-0 ceiling audit (awaiting output), then 2b environment.
> Resume from Phase ___.

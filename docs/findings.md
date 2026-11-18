# Findings log (feeds the paper's dataset-shift and design sections)

All numbers below are development diagnostics: ToN source, HistGB, 1 seed, evaluation
on every 10th row. Repeat over 5 seeds before quoting in the paper.

## 1. Flow-level baselines learn dataset-specific shortcuts

ROC-AUC, trained on ToN (diagonal = in-dataset test rows):

| features | ->unsw | ->cic | ->ton | ->bot |
|---|---|---|---|---|
| all 55 | 0.159 | 0.377 | 0.927 | 0.434 |
| drop volume | 0.176 | 0.373 | 0.928 | 0.429 |
| drop timing | 0.216 | 0.326 | 0.926 | 0.498 |
| drop pkt_size | 0.228 | 0.348 | 0.926 | 0.772 |
| drop retrans | 0.197 | 0.345 | 0.928 | 0.702 |
| drop host_fingerprint | 0.529 | 0.795 | 0.963 | 0.278 |
| drop tcp_flags | 0.178 | 0.318 | 0.927 | 0.452 |
| drop proto | 0.176 | 0.382 | 0.928 | 0.474 |
| only volume | 0.540 | 0.783 | 0.957 | 0.364 |
| only timing | 0.488 | 0.794 | 0.742 | 0.692 |
| only pkt_size | 0.631 | 0.776 | 0.942 | 0.209 |
| only retrans | 0.401 | 0.506 | 0.530 | 0.510 |
| only host_fingerprint | 0.100 | 0.605 | 0.876 | 0.856 |
| only tcp_flags | 0.654 | 0.782 | 0.856 | 0.369 |
| only proto | 0.435 | 0.711 | 0.480 | 0.429 |

Reading:
- host_fingerprint (TTL, TCP window) is a testbed shortcut: alone it ranks UNSW
  backwards (0.10) and BoT well (0.86); dropping it lifts UNSW/CIC transfer and even
  in-dataset AUC.
- ToN-IoT and BoT-IoT were, as far as we know, both built in the UNSW Canberra lab, so
  BoT is not an independent unseen domain (hypothesis, to be verified in the paper).
- Behavioural groups alone reach about 0.78 AUC on CIC but transfer to UNSW is weak.
- AUC overstates usefulness: TPR at 1% FPR on CIC stays below 0.02 even at AUC 0.80.
- The decision threshold does not transfer either: source-threshold FPR on CIC is 11%
  instead of 1%.
- Integrity: every dataset has now been inspected in diagnostics. Model and feature
  decisions use source-only leave-one-dataset-out validation from here on.

## 2. Host-window audit (10 s windows)

Collateral = benign flows lost if attacker entities are blocked permanently after their
first attack window. Preventable = share of attack flows after that first window.

| | entity | active/window median (max) | mixed | one-shot | preventable | collateral |
|---|---|---|---|---|---|---|
| unsw | host | 14 (25) | 0.0% | 0.0% | 99.7% | 0.0% |
| cic | host | 107 (221) | 0.2% | 62.7% | 99.2% | 1.8% |
| ton | host | 7 (124) | 10.8% | 10.5% | 99.6% | 54.0% |
| bot | host | 4 (11) | 6.4% | 0.5% | 98.7% | 86.4% |
| unsw | pair | 87 (106) | 0.0% | 0.0% | 99.1% | 0.0% |
| cic | pair | 227 (774) | 0.1% | 52.6% | 97.5% | 0.4% |
| ton | pair | 15 (713) | 14.1% | 8.5% | 99.0% | 49.6% |
| bot | pair | 4 (41) | 0.6% | 8.9% | 96.5% | 68.8% |
| unsw | service | 201 (300) | 0.0% | 72.9% | 82.0% | 0.0% |
| cic | service | 241 (2,395) | 0.1% | 61.8% | 97.4% | 0.2% |
| ton | service | 21 (100,421) | 0.6% | 84.4% | 93.6% | 12.5% |
| bot | service | 30 (143,079) | 0.0% | 90.0% | 89.6% | 18.3% |

Reading:
- No granularity works alone. Pair level does not fix ToN or BoT collateral because
  attacks come in bursts and the same pair is benign in other windows. Service level
  explodes (100k+ entities in one window) and is mostly one-shot.
- BoT collateral (86%) is only about 45k flows against 16.9M attack flows: rewards must be
  class-normalised (fraction of attack flows prevented, fraction of benign flows lost).
- Blocks must expire. Next audit: oracle timed blocks (`TIMED BLOCK` lines in
  `scripts/audit_hosts.py`).

## 3. Oracle timed blocks (10 s windows; attack flows prevented / benign flows lost)

Oracle: at an entity's attack window, if not blocked, block it for the next k windows.
Perfect detection and no delay, so these are ceilings, not achievable results.

| block | host unsw | host cic | host ton | host bot | pair unsw | pair cic | pair ton | pair bot |
|---|---|---|---|---|---|---|---|---|
| 10 s | 49.3 / 0.0 | 47.9 / 0.4 | 49.9 / 25.0 | 31.9 / 8.1 | 36.8 / 0.0 | 47.3 / 0.2 | 47.3 / 16.4 | 31.8 / 3.2 |
| 30 s | 74.2 / 0.0 | 72.5 / 0.6 | 75.2 / 38.3 | 57.6 / 18.4 | 59.9 / 0.0 | 71.3 / 0.2 | 70.1 / 27.3 | 56.8 / 10.7 |
| 60 s | 85.2 / 0.0 | 83.6 / 0.8 | 85.7 / 44.4 | 73.9 / 23.2 | 73.2 / 0.0 | 81.6 / 0.2 | 79.4 / 33.2 | 72.8 / 14.8 |
| 300 s | 96.4 / 0.0 | 95.0 / 1.1 | 96.6 / 51.1 | 92.6 / 45.1 | 91.8 / 0.0 | 93.0 / 0.3 | 94.2 / 42.6 | 91.0 / 32.7 |

Reading:
- Blocks are nearly free on unsw and cic; ton and bot pay real collateral.
- Returns diminish after about 60 s while collateral keeps growing (ton host: 60 s to 300 s
  adds 11 points prevented and 7 points lost).
- Pair level loses less benign traffic than host level at similar prevention.

## Provisional MDP design (pending the flow-level ceiling audit)
- Decisions per host pair per 10 s window; flows inherit the action.
- Actions with data-derived effects: monitor (none), inspect (10 s quarantine), block (60 s).
- Rewards class-normalised: fraction of attack flows prevented, fraction of benign flows lost.
- Main evaluation: flow-level detection with alert = not monitor; response metrics reported
  separately and flagged as simulated.
- Risk: flows inherit their pair-window's action, so mixed pair-windows cap flow-level
  detection (ToN: 27.7% of flows are in mixed pair-windows). The ceiling audit quantifies it.

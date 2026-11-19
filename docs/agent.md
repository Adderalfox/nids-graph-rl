# Experiment A: MLP DQN-family agent and the in-environment supervised baseline

## Agent (src/nidsrl/rl, src/nidsrl/experiments/dqn.py)
- Input: 97-number pair-window observation (docs/host_window_mdp.md) minus the dropped feature
  groups (`--drop-groups`; the mask hits both the mean and the max block). Same mask in training and
  evaluation.
- Variants: `dqn` (plain), `ddqn` (+Double), `d3qn` (+dueling). Experiments B and C reuse the exact
  RL machinery; only the encoder (graph) and the replay strategy change, so each contribution is
  isolated. Report `d3qn-mlp` as "A".
- Loss/updates: Huber, Adam, grad-clip 10, Polyak tau 0.005, `online.eval()` whenever acting,
  no-grad targets (built in numpy: `rl/targets.py`), uniform ring-buffer replay (O(1)).
- Exploration: epsilon-greedy per pair, linear decay over `eps_decay_windows` environment windows.
- Targets: y = r + gamma**dt * (1-done) * V(s2); gamma 0.9 by default.
- Model selection: source VALIDATION macro-F1 of the greedy policy, per epoch, patience 3. No target
  data is used anywhere before final evaluation.
- Ranking score for ROC/PR: max(Q_inspect, Q_block) - Q_monitor. The argmax policy is the operating
  point; thresholds do not transfer (docs/findings.md), so report both.

## Feature policy (integrity)
Main configuration drops `host_fingerprint` (a-priori group: TTL/TCP window = testbed fingerprint;
and in-dataset ToN AUC rises when it is dropped, a SOURCE-only observation). Always also report the
all-features run (`--tag allfeat`). Do not change the policy after looking at target results.

## Required controls
| control | flag | question |
|---|---|---|
| all features | no `--drop-groups`, `--tag allfeat` | does the agent exploit the shortcut? |
| contextual bandit | `--gamma 0 --tag gamma0` | does MDP structure/bootstrapping help? |
| plain DQN / Double DQN | `--variant dqn|ddqn` | RL-machinery ablation |
| supervised in same env | `scripts/run_pair_baseline.py` | what does RL add over HGB on the same obs? |

## Supervised pair baseline (experiments/pair_baseline.py)
HistGradientBoosting on the same observations, trained on pair-windows seen under a do-nothing
policy; threshold from source val at flow-weighted FPR 1%; alerts via inspect or block. Caveat: it
never saw action-history features other than "no action" during training, while at evaluation those
reflect its own actions (a mild train/test mismatch that the RL agent does not have).

## Status
All numpy logic is unit tested (replay, targets incl. Double/terminal/dt, schedules, masks,
trainer end-to-end with a linear learner, pair baseline, reports). `rl/dqn.py` and
`tests/test_dqn_torch.py` could not be run in the authoring sandbox (no torch): run
`uv run pytest tests/test_dqn_torch.py` first.

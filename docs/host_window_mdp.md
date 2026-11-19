# Host-pair response MDP (implemented in `nidsrl.rl`)

## Why host pairs, 10 s windows, 30-minute episodes
Audits (docs/findings.md) show that per-host decisions have large benign collateral on
ToN/BoT and that per-service decisions explode in size, while per-pair decisions keep a flow-level
oracle ceiling of 97-100% macro-F1. An episode is one 30-minute run of the split plan; the
purged gaps between runs guarantee train/test independence, and temporal state resets at each
run start.

## Entities, observations, actions
- Entity: (src host, dst host) pair. Group: one entity's flows in one 10 s window.
- Observation per active entity (97 numbers, identity-free): mean of the 55 scaled flow
  features, max of the 35 continuous ones, log flow count, and history features keyed by the
  stable host-pair id: previous log count, its change, log windows seen, last action
  inspect/block flags, time since a hold ended.
- Actions: 0 monitor; 1 inspect = pair's flows are held (removed) for 1 window; 2 block = held
  for 6 windows. Held flows never reach the agent, so actions change the next state. Effects
  are a replay-based simulation on recorded flows, not real counterfactual traffic.
- Ground-truth attack/benign counts feed rewards and evaluation only, never observations
  (`test_observation_never_depends_on_labels`).

## Reward
Per acted entity-window, class-balanced weights u_a = 0.5/p_attack, u_b = 0.5/p_benign
(source train only):

    flow term  alert:    det_gain*u_a*A - fp_cost*u_b*B
               no alert: -miss_cost*u_a*A
    response   alert:    prevent_gain*u_a*A_held - collateral_cost*u_b*B_held
    r = sign(x)*log1p(|x|) - action_cost[a]

A, B are the attack and benign flows of the entity in the window; A_held, B_held those in the
held windows that follow. The log compresses heavy-tailed flow counts. Weights are
hyperparameters chosen on SOURCE validation only; they are assumptions and need a
sensitivity analysis in the paper.

## Evaluation (comparable with the supervised baselines)
Every flow inherits the alert state of its entity-window: alerted if the pair's action in that
window was not monitor, or the pair was held when the flow occurred. Held attack flows count as
detected, held benign flows as false alarms. The ranking score is the agent's attack score
(for DQN: max(Q_inspect, Q_block) - Q_monitor); held flows inherit the score at the time of the
hold. Response metrics are reported separately: prevented attack share, benign collateral
share, alert share, number of blocks and inspects.

## Validation tools
- `scripts/check_mdp.py`: runs never / oracle policies. The oracle any-attack block policy
  must reproduce the TIMED BLOCK 60 s line of `audit_hosts.py --granularity pair` within about
  a point (`test_env_matches_hostaudit_timed_block_oracle` checks the equivalence exactly).
- Oracle policies read ground truth and exist only for diagnostics.

## Known limitations
- Replay-based consequences; attackers cannot adapt to blocks.
- One 30-minute run per episode: no state carries across runs.
- Pair-window decisions cannot split mixed flows (ceiling 97.3% on ToN).

## Transitions and discounting (added in 2b-2)
`observe()` returns, per pair that reappears, the transition (s, a, r, s2, done=False, dt) where dt
is the number of windows since that pair's last decision (a block of 6 windows gives dt >= 7).
Targets use gamma ** dt (semi-MDP). Pairs with no further decision in the run get a terminal
transition (done=True, dt=1). Because the hold consequence (A_held, B_held) is already inside the
reward, gamma mainly carries "this pair keeps coming back" information; `--gamma 0` (contextual
bandit) is therefore a required control to show the MDP structure matters.

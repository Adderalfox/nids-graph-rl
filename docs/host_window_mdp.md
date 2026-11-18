# Host-window response MDP (draft; finalized after the Phase 2b-0 audit)

## Idea
At each 10 s window the agent sees the temporal host graph and chooses, for every
active host, one of: monitor, inspect, block. A shared Q-head outputs 3 values per
node (factored per-node Q-learning with shared weights); the joint action is the
per-host choice for the window.

## It is a SIMULATED response environment
The datasets are recorded traffic. No real counterfactual exists: we cannot observe
what an attacker would have done after being blocked. Effects are therefore simulated
by replaying the recording with the acted-on host's later flows modified:
- monitor: no change.
- inspect (proposal): throttle / divert for deep inspection: a fraction of the host's
  next-window flows is held (attack flows mitigated with probability rho_a, benign
  flows delayed or dropped with probability rho_b < rho_a); small per-action cost;
  limited inspection budget per window (analyst capacity).
- block: the host's flows are removed from all later windows of the run.

Rewards use recorded labels: damage prevented (attack flows or bytes removed), benign
collateral (benign flows removed), action costs. Cost parameters are assumptions;
the paper must say so and include a sensitivity analysis.

## Why this gives actions real consequences
Blocking or throttling changes the next window's graph (the host's edges disappear),
so the transition depends on the action. That is what makes it an MDP rather than a
contextual bandit.

## Evaluation (comparable to baselines)
For flow-level metrics, a flow is "alerted" if its source host's action in that
window is not monitor. The continuous score for ROC metrics is
max(Q(inspect), Q(block)) - Q(monitor). No label information from inspection may be
visible to the agent at test time. Response metrics (attack flows prevented, benign
collateral, inspections used) are reported separately and are parameter dependent.

## Realism checks (see scripts/audit_hosts.py)
1. Active source hosts per window: bounds the action space.
2. Host-window purity: share of host-windows that mix benign and attack flows.
3. Attacker persistence: if attackers are one-shot, blocking prevents nothing.
4. Preventable attack-flow share and benign collateral: the achievable reward range.

## Open questions after the audit
- Host-level label rule for mixed hosts.
- Cap on hosts per window (top-K by flow count) for the 4 GB GPU.
- Whether `inspect` earns its place or collapses into a hedged block.

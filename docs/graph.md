# Graph encoder (Phase 3) and experiment G

Window graph: nodes = hosts active in the 10 s window, edges = active host pairs (the env's
decision units). Edge features = the pair-window observation (feature-policy mask applied).
Node features (identity-free): log out/in degree and log out/in flow volume. Hosts are re-indexed
locally per window; no host ID, port or address reaches the model (ADR 0003).

`models/gat.py::EdgeGAT`: edge/node encoders, `layers` GATv2-style attention layers over messages
in both directions (edge features inside the score, learned direction embedding, residual +
LayerNorm), head on [h_src, h_dst, edge embedding]. `layers=0` is the edge-MLP ablation.
Pure torch (no torch_geometric): attention softmax via scatter_reduce/index_add.

Experiment G (`scripts/train_gat.py`): supervised edge classifier, same labels, weights,
threshold rule, environment and metrics as the pair-level gradient-boosting baseline. It answers
"does the graph help, with no RL involved?". Next: the same EdgeGAT with 3 dueling outputs as the
Q-network of the DQN trainer (experiment B), using window-snapshot replay.

Status: numpy parts (snapshot, collate, batching, trainer, policy) are unit-tested; `gat.py` is
untested in the authoring sandbox (no torch): run `uv run pytest tests/test_gat_torch.py` first.

## Experiment B: EdgeGAT as the D3QN Q-network (`scripts/train_gdqn.py`)
Same env, reward, targets (numpy `td_targets`, gamma**dt, Double DQN), feature policy and
evaluation as experiment A. Replay stores whole window snapshots per run (`rl/graph_replay.py`);
a batch is a few random windows with all their transitions plus the snapshots where those pairs
next appeared. Env transitions carry (t0, i0, t1, i1) pointers. Oldest runs are evicted first.
`--layers 0` is the no-message-passing ablation under identical RL machinery.

## Reporting: raw argmax and calibrated operating point
Q-agents are reported twice: `*-d3qn` (argmax policy, wherever the reward puts the operating
point) and `*-d3qn-cal` (alert when attack_score(Q) exceeds a SOURCE-validation threshold at a
flow-weighted FPR budget; alert type inspect/block by Q). The calibrated rows are the ones to
compare with the supervised baselines.

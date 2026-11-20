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

import numpy as np
from rl_helpers import LinearScorer

from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.experiments.gat_supervised import (
    collect_snap_run,
    make_batches,
    train_graph_supervised,
    val_threshold,
)
from nidsrl.graph.snapshot import LOGN_COL, NODE_DIM, build_snapshot, collate
from nidsrl.rl.env import OBS_DIM, EnvConfig, HostPairEnv, RewardConfig, class_weights
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import GraphThresholdPolicy
from nidsrl.rl.rollout import evaluate_loaded_runs
from nidsrl.rl.rundata import load_run, run_ids
from nidsrl.rl.trainer import RunCache


def _obs(flows):
    o = np.zeros((len(flows), OBS_DIM), dtype=np.float32)
    o[:, LOGN_COL] = np.log1p(np.asarray(flows, dtype=np.float32)) / 10.0
    return o


def test_snapshot_nodes_edges_and_features_are_identity_free():
    # edges: 10->20 (3 flows), 10->30 (5 flows), 20->30 (1 flow); host ids are arbitrary
    obs = _obs([3, 5, 1])
    snap = build_snapshot(obs, np.array([10, 10, 20]), np.array([20, 30, 30]))
    assert snap.n_nodes == 3 and snap.n_edges == 3 and snap.node_x.shape == (3, NODE_DIM)
    # relabelling hosts must not change features up to node order
    snap2 = build_snapshot(obs, np.array([900, 900, 5]), np.array([5, 77, 77]))
    key = lambda s: sorted(map(tuple, np.round(s.node_x, 5)))  # noqa: E731
    assert key(snap) == key(snap2)
    # host 10 sends 2 edges / 8 flows: log1p(2)/5, log1p(8)/12
    n10 = snap.src[0]
    assert np.isclose(snap.node_x[n10, 0], np.log1p(2) / 5, atol=1e-4)
    assert np.isclose(snap.node_x[n10, 2], np.log1p(8) / 12, atol=1e-3)


def test_snapshot_mask_and_empty():
    obs = _obs([2, 2])
    keep = obs_keep_indices(drop_groups=("host_fingerprint",))
    assert build_snapshot(obs, np.array([1, 2]), np.array([2, 3]), keep).x.shape == (2, len(keep))
    e = build_snapshot(np.zeros((0, OBS_DIM), np.float32), np.zeros(0, int), np.zeros(0, int))
    assert e.n_edges == 0 and e.n_nodes == 0


def test_collate_offsets_keep_graphs_disjoint():
    a = build_snapshot(_obs([1, 1]), np.array([1, 2]), np.array([2, 3]))
    b = build_snapshot(_obs([1]), np.array([7]), np.array([8]))
    g = collate([a, b])
    assert g.edge_ptr.tolist() == [0, 2, 3] and g.node_ptr.tolist() == [0, 3, 5]
    assert g.src[2] >= 3 and g.dst[2] >= 3  # second graph's nodes are offset
    assert g.src[:2].max() < 3 and len(g.x) == 3 and len(g.node_x) == 5


def _setup(world):
    root, _, _ = world
    plan = load_plan(root / "splits" / "a.json")
    scaler = load_scaler(root / "scalers" / "a.json")
    ua, ub = class_weights(plan)
    cfg = EnvConfig(reward=RewardConfig(u_a=ua, u_b=ub))
    keep = obs_keep_indices()
    runs = {
        s: [load_run(root / "a", plan, s, r, scaler) for r in run_ids(root / "a", plan, s)]
        for s in ("train", "val")
    }
    return root, plan, scaler, cfg, keep, runs, ua, ub


def test_collect_and_batches_have_consistent_labels(world):
    root, plan, scaler, cfg, keep, runs, ua, ub = _setup(world)
    sr = collect_snap_run(HostPairEnv(cfg), runs["train"][0], keep)
    assert len(sr.snaps) > 0 and all(s.x.shape[1] == len(keep) for s in sr.snaps)
    n_edges = sum(s.n_edges for s in sr.snaps)
    batches = make_batches(sr, 5, ua, ub, np.random.default_rng(0))
    assert sum(len(y) for _, y, _, _ in batches) == n_edges
    for g, y, w, ben in batches:
        assert len(y) == len(w) == len(ben) == len(g.x) and (w > 0).all()
        assert set(np.unique(y)) <= {0.0, 1.0}


def test_graph_supervised_trainer_and_threshold_policy_end_to_end(world):
    root, plan, scaler, cfg, keep, runs, ua, ub = _setup(world)
    cache = RunCache(
        lambda i: collect_snap_run(HostPairEnv(cfg), runs["train"][i], keep),
        size_fn=lambda sr: sr.nbytes(),
    )
    val = [collect_snap_run(HostPairEnv(cfg), r, keep) for r in runs["val"]]
    sc = LinearScorer(len(keep))
    hist = train_graph_supervised(
        sc, list(range(len(runs["train"]))), cache.get, val, ua, ub,
        epochs=6, windows_per_batch=8, patience=0, seed=0, log=lambda *_: None,
    )  # fmt: skip
    assert hist[-1]["val_loss"] < hist[0]["val_loss"]
    thr, q = val_threshold(sc, val, ua, ub, 0.05)
    pol = GraphThresholdPolicy(sc.score, thr, 2, keep)
    rep = evaluate_loaded_runs(pol, runs["val"], env_cfg=cfg, min_group_flows=50)
    assert rep["binary"]["tpr"] > 0.8 and rep["binary"]["fpr"] < 0.3

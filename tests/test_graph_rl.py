import numpy as np
from rl_helpers import LinearGraphLearner
from test_graph_core import _setup

from nidsrl.experiments.calibration import q_threshold_flat, q_threshold_graph
from nidsrl.experiments.gat_supervised import collect_snap_run
from nidsrl.graph.snapshot import Snapshot
from nidsrl.rl.env import HostPairEnv
from nidsrl.rl.graph_replay import GraphReplay, GraphRunBuilder
from nidsrl.rl.graph_trainer import batch_targets, train_graph_dqn
from nidsrl.rl.policy import GraphQThresholdPolicy, QThresholdPolicy
from nidsrl.rl.rollout import NeverPolicy, evaluate_loaded_runs
from nidsrl.rl.trainer import TrainConfig


def _snap(tag, n_edges, d=3):
    """Edge row r of snapshot `tag` has feature vector [tag, r, 0] (identifiable)."""
    x = np.zeros((n_edges, d), dtype=np.float32)
    x[:, 0], x[:, 1] = tag, np.arange(n_edges)
    z = np.zeros(n_edges, dtype=np.int64)
    return Snapshot(x, z, z + 1, np.zeros((2, 4), dtype=np.float32))


def test_env_transition_pointers_address_the_right_rows(world):
    root, plan, scaler, cfg, keep, runs, ua, ub = _setup(world)
    env = HostPairEnv(cfg)
    run = runs["train"][0]
    env.reset(run)
    seen: dict[int, np.ndarray] = {}
    checked = 0
    while not env.done:
        obs = env.observe()
        seen[obs["t"]] = obs["obs"].copy()
        tr = obs["transitions"]
        if tr is not None:
            for k in range(len(tr["a"])):
                assert np.allclose(seen[int(tr["t0"][k])][tr["i0"][k]], tr["s"][k])
                assert np.allclose(obs["obs"][tr["i1"][k]], tr["s2"][k])
                checked += 1
        n = len(obs["gids"])
        env.step(np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.float32))
    final = env.finish().final_transitions
    for k in range(len(final["a"])):
        assert np.allclose(seen[int(final["t0"][k])][final["i0"][k]], final["s"][k])
    assert checked > 0 and len(final["a"]) > 0


def test_replay_batch_gathers_state_and_next_state_rows():
    b = GraphRunBuilder()
    b.add_snapshot(0, _snap(10, 3))
    b.add_snapshot(1, _snap(11, 2))
    b.add_snapshot(5, _snap(12, 4))
    # edge 2 of window 0 -> edge 1 of window 5 (dt 5); edge 0 of window 1 -> edge 3 of window 5
    b.add_transitions({"t0": np.array([0, 1]), "i0": np.array([2, 0]), "t1": np.array([5, 5]),
                       "i1": np.array([1, 3]), "a": np.array([2, 1]), "r": np.array([1.0, 2.0]),
                       "done": np.array([False, False]), "dt": np.array([5, 4])})  # fmt: skip
    b.add_transitions({"t0": np.array([5]), "i0": np.array([0]), "t1": np.array([5]),
                       "i1": np.array([0]), "a": np.array([0]), "r": np.array([3.0]),
                       "done": np.array([True]), "dt": np.array([1])}, final=True)  # fmt: skip
    run = b.build()
    rep = GraphReplay(100, seed=0)
    rep.add(run)
    for _ in range(20):
        u = rep.sample(2)
        s_feat = u.graph.x[u.rows]  # [tag, row]
        for k in range(len(u.a)):
            if u.done[k]:
                assert s_feat[k][0] == 12 and s_feat[k][1] == 0
            elif u.a[k] == 2:
                assert tuple(s_feat[k][:2]) == (10, 2)
                assert tuple(u.next_graph.x[u.next_rows[k]][:2]) == (12, 1)
            else:
                assert tuple(s_feat[k][:2]) == (11, 0)
                assert tuple(u.next_graph.x[u.next_rows[k]][:2]) == (12, 3)


def test_replay_evicts_oldest_runs_and_targets_use_gathered_values():
    rep = GraphReplay(5, seed=0)
    for tag in (1, 2, 3):
        b = GraphRunBuilder()
        b.add_snapshot(0, _snap(tag, 2))
        z = np.zeros(3, int)
        b.add_transitions(
            {"t0": z, "i0": np.array([0, 1, 0]), "t1": z, "i1": z, "a": z, "r": np.ones(3),
             "done": np.ones(3, bool), "dt": np.ones(3, int)},
            final=True,
        )
        rep.add(b.build())
    assert len(rep.runs) == 1 and len(rep) == 3  # two oldest dropped to fit capacity 5

    # TD target through a learner whose Q(s2) is known: q_online = q_target = x[:, :3] columns
    class L:
        def next_q(self, g):
            q = np.tile(g.x[:, :1], (1, 3)).astype(np.float32)
            return q, q

    b = GraphRunBuilder()
    b.add_snapshot(0, _snap(0, 1))
    b.add_snapshot(1, _snap(7, 2))
    b.add_transitions({"t0": np.array([0]), "i0": np.array([0]), "t1": np.array([1]),
                       "i1": np.array([1]), "a": np.array([1]), "r": np.array([2.0]),
                       "done": np.array([False]), "dt": np.array([2])})  # fmt: skip
    r2 = GraphReplay(10)
    r2.add(b.build())
    y = batch_targets(L(), r2.sample(1), gamma=0.5, double=True)
    assert np.allclose(y, 2.0 + 0.5**2 * 7.0)


def test_graph_dqn_trainer_learns_end_to_end(world):
    root, plan, scaler, cfg, keep, runs, ua, ub = _setup(world)
    learner = LinearGraphLearner(len(keep))
    tc = TrainConfig(epochs=6, learn_start=300, buffer_capacity=40_000, eps_decay_windows=300,
                     updates_per_window=2, patience=0)  # fmt: skip
    hist = train_graph_dqn(
        learner, HostPairEnv(cfg), list(range(len(runs["train"]))), lambda i: runs["train"][i],
        tc, keep=keep, windows_per_update=4, seed=0, log=lambda *_: None,
        validate=lambda pol: evaluate_loaded_runs(
            pol, runs["val"], env_cfg=cfg, min_group_flows=50
        ),
    )  # fmt: skip
    assert learner.fits > 50 and hist[-1]["updates"] == learner.fits
    val = max(h["val"]["metric"] for h in hist if "val" in h)
    base = evaluate_loaded_runs(NeverPolicy(), runs["val"], env_cfg=cfg, min_group_flows=50)
    assert val > base["binary"]["macro_f1"] + 0.2


def test_q_calibration_and_threshold_policies(world):
    root, plan, scaler, cfg, keep, runs, ua, ub = _setup(world)
    w = np.zeros((len(keep), 3))
    w[0, 2] = 1.0  # Q(block) = feature 0; attack_score = feature 0 (log_in_bytes-like)
    q_flat = lambda x: x @ w  # noqa: E731
    q_graph = lambda s: s.x @ w  # noqa: E731
    thr = q_threshold_flat(q_flat, runs["val"], keep, cfg, 0.05)
    snaps = [collect_snap_run(HostPairEnv(cfg), r, keep) for r in runs["val"]]
    thr_g = q_threshold_graph(q_graph, snaps, 0.05)
    assert np.isfinite(thr) and np.isfinite(thr_g)
    flat = QThresholdPolicy(q_flat, thr, keep)  # keep is the full mask here
    graph = GraphQThresholdPolicy(q_graph, thr_g, keep)
    for pol in (flat, graph):
        rep = evaluate_loaded_runs(pol, runs["val"], env_cfg=cfg, min_group_flows=50)
        assert rep["binary"]["fpr"] < 0.5 and rep["binary"]["tpr"] > 0.5

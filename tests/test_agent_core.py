import numpy as np
import pytest
from rl_helpers import LinearLearner
from test_env import SCENARIO, make_run

from nidsrl.data.scaling import load_scaler
from nidsrl.data.splits import load_plan
from nidsrl.rl.env import BLOCK, OBS_DIM, EnvConfig, HostPairEnv, RewardConfig, class_weights
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import QPolicy, ThresholdPolicy
from nidsrl.rl.replay import ReplayBuffer
from nidsrl.rl.rollout import NeverPolicy, evaluate_loaded_runs
from nidsrl.rl.rundata import N_FEATURES, load_run, run_ids
from nidsrl.rl.targets import attack_score, epsilon, select_actions, td_targets
from nidsrl.rl.trainer import RunCache, TrainConfig, train_dqn


def tr(n, start=0, obs_dim=4):
    v = np.arange(start, start + n, dtype=np.float32)
    return {
        "s": np.repeat(v[:, None], obs_dim, 1),
        "a": (v % 3).astype(np.int64),
        "r": v,
        "s2": np.repeat(v[:, None] + 0.5, obs_dim, 1),
        "done": v % 2 == 0,
        "dt": np.ones(n, dtype=np.int64),
    }


def test_replay_ring_overwrites_oldest():
    b = ReplayBuffer(5, 4)
    b.add(tr(3))
    b.add(tr(4, start=10))
    assert len(b) == 5
    assert sorted(b.r.tolist()) == [2.0, 10.0, 11.0, 12.0, 13.0]
    s = b.sample(50)
    assert set(s["r"].tolist()) <= {2.0, 10.0, 11.0, 12.0, 13.0}
    assert (s["s"][:, 0] == s["r"]).all()  # fields stay aligned


def test_replay_oversized_batch_keeps_newest():
    b = ReplayBuffer(3, 4)
    b.add(tr(10))
    assert sorted(b.r.tolist()) == [7.0, 8.0, 9.0]


def test_td_targets_dqn_double_terminal_and_dt():
    r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    done = np.array([False, True, False])
    dt = np.array([1, 1, 3])
    q_t = np.array([[0, 4, 2], [0, 4, 2], [0, 4, 2]], dtype=np.float32)
    q_o = np.array([[0, 1, 5]] * 3, dtype=np.float32)
    dqn = td_targets(r, done, dt, 0.5, q_t)
    assert np.allclose(dqn, [1 + 0.5 * 4, 1.0, 1 + 0.5**3 * 4])
    dbl = td_targets(r, done, dt, 0.5, q_t, q_o)  # online argmax = action 2 -> target value 2
    assert np.allclose(dbl, [1 + 0.5 * 2, 1.0, 1 + 0.5**3 * 2])
    assert np.allclose(td_targets(r, done, dt, 0.0, q_t), r)  # gamma 0 = bandit control


def test_epsilon_schedule_and_selection():
    assert epsilon(0, 1.0, 0.1, 100) == 1.0
    assert epsilon(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert epsilon(500, 1.0, 0.1, 100) == pytest.approx(0.1)
    q = np.tile(np.array([0.0, 3.0, 1.0]), (2000, 1))
    rng = np.random.default_rng(0)
    assert (select_actions(q, 0.0, rng) == 1).all()
    share = np.bincount(select_actions(q, 1.0, rng), minlength=3) / 2000
    assert (share > 0.25).all()
    assert attack_score(q[:1]).tolist() == [3.0]


def test_obs_mask_drops_group_in_mean_and_max_blocks():
    full = obs_keep_indices()
    assert len(full) == OBS_DIM
    k = obs_keep_indices(drop_groups=("host_fingerprint",))
    assert len(k) == OBS_DIM - 8  # 4 features removed from the mean and the max block
    assert len(obs_keep_indices(keep_groups=("volume",))) == 6 + 6 + 7


def test_env_transitions_carry_dt():
    env = HostPairEnv(EnvConfig(inspect_windows=1, block_windows=2, reward=RewardConfig()))
    run = make_run(SCENARIO, n_windows=8)
    env.reset(run)
    dts = []
    while not env.done:
        obs = env.observe()
        if obs["transitions"] is not None:
            dts += obs["transitions"]["dt"].tolist()
        env.step(np.where(env.current_labels()[0] > 0, BLOCK, 0))
    assert dts and min(dts) >= 1
    assert max(dts) == 3  # attacker blocked 2 windows -> next decision 3 windows later
    assert (env.finish().final_transitions["dt"] == 1).all()


def test_policies_apply_mask_and_handle_empty():
    keep = np.array([0, 2])
    seen = {}

    def qfn(x):
        seen["shape"] = x.shape
        return np.tile([0.0, 1.0, 0.0], (len(x), 1))

    p = QPolicy(qfn, keep)
    a, s = p.act({"obs": np.ones((5, OBS_DIM), dtype=np.float32)}, None)
    assert seen["shape"] == (5, 2) and (a == 1).all() and (s == 1).all()
    a, s = p.act({"obs": np.zeros((0, OBS_DIM), dtype=np.float32)}, None)
    assert len(a) == 0
    t = ThresholdPolicy(lambda x: x[:, 0], 0.5, BLOCK)
    a, _ = t.act({"obs": np.array([[0.1] * 3, [0.9] * 3], dtype=np.float32)}, None)
    assert a.tolist() == [0, BLOCK]


def test_run_cache_respects_budget_and_reuses():
    calls = []

    def loader(i):
        calls.append(i)
        return make_run(SCENARIO, n_windows=6)

    c = RunCache(loader, max_bytes=1)
    c.get(1)
    c.get(1)
    assert calls == [1]
    c.get(2)
    assert len(c._store) == 1  # budget evicts older runs


def test_trainer_learns_on_synthetic_data_end_to_end(world):
    root, plans, pairs = world
    scaler = load_scaler(root / "scalers" / "a.json")
    plan = load_plan(root / "splits" / "a.json")
    ids = run_ids(root / "a", plan, "train")
    val_ids = run_ids(root / "a", plan, "val")
    cache = RunCache(lambda r: load_run(root / "a", plan, "train", r, scaler))
    vruns = [load_run(root / "a", plan, "val", r, scaler) for r in val_ids]
    ua, ub = class_weights(plan)
    env_cfg = EnvConfig(reward=RewardConfig(u_a=ua, u_b=ub))
    env = HostPairEnv(env_cfg)
    keep = obs_keep_indices()
    learner = LinearLearner(len(keep), lr=0.05)
    cfg = TrainConfig(epochs=6, learn_start=300, batch_size=128, buffer_capacity=50_000,
                      eps_decay_windows=300, patience=0)  # fmt: skip
    best = {}
    hist = train_dqn(
        learner, env, ids, cache.get, cfg, obs_dim=len(keep), keep=keep, seed=0,
        validate=lambda pol: evaluate_loaded_runs(pol, vruns, env_cfg=env_cfg, min_group_flows=50),
        on_best=lambda rec: best.update(W=learner.W.copy(), rec=rec), log=lambda *_: None,
    )  # fmt: skip
    assert learner.fits > 100 and hist[-1]["updates"] == learner.fits
    assert any(h.get("best") for h in hist)
    val = max(h["val"]["metric"] for h in hist if "val" in h)
    base = evaluate_loaded_runs(NeverPolicy(), vruns, env_cfg=env_cfg, min_group_flows=50)
    assert val > base["binary"]["macro_f1"] + 0.2  # the agent learned to alert on attacks


def test_trainer_stops_at_max_updates_and_is_seeded(world):
    root, plans, _ = world
    scaler = load_scaler(root / "scalers" / "a.json")
    plan = load_plan(root / "splits" / "a.json")
    ids = run_ids(root / "a", plan, "train")
    cache = RunCache(lambda r: load_run(root / "a", plan, "train", r, scaler))
    out = []
    for _ in range(2):
        keep = obs_keep_indices()
        lr = LinearLearner(len(keep))
        cfg = TrainConfig(epochs=3, max_updates=50, learn_start=100, batch_size=64,
                          buffer_capacity=10_000, patience=0)  # fmt: skip
        train_dqn(lr, HostPairEnv(), ids, cache.get, cfg, obs_dim=len(keep), keep=keep,
                  seed=3, log=lambda *_: None)  # fmt: skip
        assert lr.fits == 50
        out.append(lr.W.copy())
    assert np.allclose(out[0], out[1])


def test_n_features_constant():
    assert N_FEATURES == 55

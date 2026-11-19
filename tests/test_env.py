import numpy as np
import pandas as pd
import pytest

from nidsrl.data.hostaudit import _timed_block_sim
from nidsrl.rl.env import (
    BLOCK,
    INSPECT,
    OBS_DIM,
    EnvConfig,
    HostPairEnv,
    RewardConfig,
    class_weights,
)
from nidsrl.rl.rollout import NeverPolicy, OraclePolicy, rollout_run, summarize_results
from nidsrl.rl.rundata import N_FEATURES, build_groups


def make_run(flows, n_windows=10, seed=0):
    """flows: list of (window, src, dst, attack_flows, benign_flows)."""
    rng = np.random.default_rng(seed)
    win, src, dst, y = [], [], [], []
    for w, s, d, a, b in flows:
        for label, count in ((1, a), (0, b)):
            win += [w] * count
            src += [s] * count
            dst += [d] * count
            y += [label] * count
    n = len(win)
    x = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    cat = np.where(np.array(y) == 1, 1, 0).astype(np.int8)
    return build_groups(
        0,
        np.array(win),
        np.array(src),
        np.array(dst),
        x,
        np.array(y, dtype=np.uint8),
        cat,
        n_windows,
    )


# pair 1->9 attacks in windows 0..3 (5 attack + 2 benign each); pair 2->9 benign only (4 per window)
SCENARIO = [(w, 1, 9, 5, 2) for w in range(4)] + [(w, 2, 9, 0, 4) for w in range(4)]


def cfg(block=2, inspect=1):
    return EnvConfig(inspect_windows=inspect, block_windows=block, reward=RewardConfig())


def test_build_groups_counts_and_pointers():
    run = make_run(SCENARIO, n_windows=6)
    assert run.n_groups == 8 and run.n_pairs == 2
    assert run.g_win.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert run.win_ptr.tolist() == [0, 2, 4, 6, 8, 8, 8]
    assert int(run.g_A.sum()) == 20 and int(run.g_B.sum()) == 4 * 2 + 16
    assert run.g_cats[:, 1].sum() == 20 and run.g_cats[:, 0].sum() == 24
    # order_pw sorts by (pair, window): inverse permutation is consistent
    assert (run.order_pw[run.pos_in_pw] == np.arange(run.n_groups)).all()


def test_block_removes_future_flows_and_is_scored_correctly():
    run = make_run(SCENARIO, n_windows=6)
    env = HostPairEnv(cfg(block=2))
    env.reset(run)
    obs = env.observe()
    assert obs["obs"].shape == (2, OBS_DIM)
    # block pair 1 (the attacker), monitor pair 2
    attacker = run.g_pair[obs["gids"]] == run.g_pair[obs["gids"]][np.argmax(run.g_A[obs["gids"]])]
    actions = np.where(attacker, BLOCK, 0)
    rewards = env.step(actions)
    # flow term 5 - 2 = 3; held windows 1 and 2: 10 attack, 4 benign -> resp 6; x = 9
    assert rewards[attacker][0] == pytest.approx(np.log1p(9) - 0.1, abs=1e-5)
    assert rewards[~attacker][0] == 0.0  # monitor a benign pair earns nothing

    obs1 = env.observe()  # window 1: attacker is held, only the benign pair is visible
    assert len(obs1["gids"]) == 1 and run.g_A[obs1["gids"][0]] == 0
    env.step(np.zeros(1, dtype=np.int64))
    obs2 = env.observe()  # window 2: still held
    assert len(obs2["gids"]) == 1
    env.step(np.zeros(1, dtype=np.int64))
    obs3 = env.observe()  # window 3: released, both visible, transition emitted for the attacker
    assert len(obs3["gids"]) == 2
    tr = obs3["transitions"]
    assert tr is not None and (tr["a"] == BLOCK).any()
    assert tr["s"].shape[1] == OBS_DIM and not tr["done"].any()
    env.step(np.zeros(2, dtype=np.int64))
    while not env.done:
        env.observe()
        env.step(np.zeros(0, dtype=np.int64))
    res = env.finish()
    assert res.held.sum() == 2  # attacker groups at windows 1 and 2
    assert (res.A * res.held).sum() == 10 and (res.B * res.held).sum() == 4
    assert res.n_block == 1 and res.n_inspect == 0


def test_inspect_holds_one_window():
    run = make_run(SCENARIO, n_windows=6)
    res = rollout_run(HostPairEnv(cfg(inspect=1)), run, OraclePolicy("any", INSPECT))
    # attacker triggers at w0, held w1, triggers again w2, held w3 -> 2 held windows
    assert res.held.sum() == 2 and res.n_inspect == 2


def test_wrong_action_count_raises():
    run = make_run(SCENARIO, n_windows=6)
    env = HostPairEnv(cfg())
    env.reset(run)
    env.observe()
    with pytest.raises(ValueError):
        env.step(np.zeros(5, dtype=np.int64))


def test_observation_never_depends_on_labels():
    a = make_run(SCENARIO, n_windows=6, seed=3)
    b = make_run([(w, s, d, 0, x + y) for w, s, d, x, y in SCENARIO], n_windows=6, seed=3)
    assert a.g_n.tolist() == b.g_n.tolist()
    env_a, env_b = HostPairEnv(cfg()), HostPairEnv(cfg())
    env_a.reset(a)
    env_b.reset(b)
    assert np.array_equal(env_a.observe()["obs"], env_b.observe()["obs"])


def test_final_transitions_are_terminal():
    run = make_run(SCENARIO, n_windows=6)
    env = HostPairEnv(cfg())
    env.reset(run)
    collected = []
    res = rollout_run(env, run, NeverPolicy(), on_transitions=collected.append)
    last = collected[-1]
    assert last["done"].all() and len(last["a"]) == 2
    assert res.alert.sum() == 0


def test_env_matches_hostaudit_timed_block_oracle():
    """The env's any-attack oracle must reproduce the audit's timed-block simulation."""
    rng = np.random.default_rng(1)
    flows = []
    for pair in range(40):
        for w in range(60):
            if rng.random() < 0.5:
                attacker = pair < 12 and 10 <= w <= 40 and rng.random() < 0.7
                flows.append(
                    (
                        w,
                        pair,
                        100,
                        int(rng.integers(1, 5)) if attacker else 0,
                        int(rng.integers(1, 4)),
                    )
                )
    run = make_run(flows, n_windows=180)
    k = 6
    res = rollout_run(HostPairEnv(cfg(block=k)), run, OraclePolicy("any", BLOCK))
    g = pd.DataFrame({"w": run.g_win, "e": run.g_pair, "a": run.g_A, "b": run.g_B})
    prevented, collateral, blocks = _timed_block_sim(g, (k,))[k]
    assert int((res.A * res.held).sum()) == prevented
    assert int((res.B * res.held).sum()) == collateral
    assert res.n_block == blocks


def test_summarize_matches_expanded_flow_metrics():
    run = make_run(SCENARIO, n_windows=6)
    res = rollout_run(HostPairEnv(cfg(block=2)), run, OraclePolicy("any", BLOCK))
    s = summarize_results([res], shared=["dos"], unseen=[], min_group_flows=1)
    # expand to flows by hand
    A, B, alert = res.A, res.B, res.alert
    tp, fp, fn = (A * alert).sum(), (B * alert).sum(), (A * ~alert).sum()
    assert s["binary"]["tp"] == tp and s["binary"]["fp"] == fp and s["binary"]["fn"] == fn
    assert s["n_flows"] == (A + B).sum()
    assert s["response"]["prevented_attack_share"] == pytest.approx(10 / 20)


def test_class_weights_are_balanced():
    plan = {"stats": {"train": {"flows": 1000, "attacks": 50}}}
    u_a, u_b = class_weights(plan)
    assert u_a == pytest.approx(0.5 / 0.05) and u_b == pytest.approx(0.5 / 0.95)
    assert u_a * 50 == pytest.approx(u_b * 950)  # both classes carry equal total weight

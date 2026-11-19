"""Torch backend tests. Skipped when torch is not installed (e.g. CI without the gpu extra).

These run for the first time on the author's machine: run `uv run pytest tests/test_dqn_torch.py`.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nidsrl.rl.dqn import QNet, TorchLearner  # noqa: E402
from nidsrl.rl.env import OBS_DIM  # noqa: E402
from nidsrl.rl.targets import td_targets  # noqa: E402


def test_dueling_q_has_zero_mean_advantage_and_shapes():
    net = QNet(OBS_DIM, (16,), dueling=True)
    x = torch.randn(5, OBS_DIM)
    z = net.body(x)
    q = net(x)
    assert q.shape == (5, 3)
    assert torch.allclose(q.mean(1), net.value(z).squeeze(1), atol=1e-5)
    assert QNet(OBS_DIM, (16,), dueling=False)(x).shape == (5, 3)


def test_learner_fits_a_fixed_target_and_target_net_tracks_slowly():
    torch.manual_seed(0)
    lr = TorchLearner(8, (32,), dueling=True, lr=1e-2, tau=0.1)
    rng = np.random.default_rng(0)
    s = rng.normal(size=(256, 8)).astype(np.float32)
    a = rng.integers(0, 3, 256)
    y = (s[:, 0] * (a - 1)).astype(np.float32)
    first = lr.fit(s, a, y)["loss"]
    for _ in range(300):
        info = lr.fit(s, a, y)
    assert info["loss"] < 0.3 * first
    assert info["td_error"].shape == (256,)
    q_on, q_tg = lr.next_q(s)
    assert q_on.shape == q_tg.shape == (256, 3)
    assert np.abs(q_on - q_tg).mean() > 0  # Polyak target lags the online net


def test_q_values_does_not_change_weights_and_handles_empty():
    lr = TorchLearner(8, (16,))
    before = [p.clone() for p in lr.online.parameters()]
    assert lr.q_values(np.zeros((0, 8), dtype=np.float32)).shape == (0, 3)
    lr.q_values(np.ones((4, 8), dtype=np.float32))
    assert all(torch.equal(b, p) for b, p in zip(before, lr.online.parameters(), strict=True))
    assert not lr.online.training


def test_state_roundtrip_and_numpy_targets_work_with_torch_outputs():
    a, b = TorchLearner(8, (16,)), TorchLearner(8, (16,))
    b.load(a.state())
    x = np.random.default_rng(1).normal(size=(6, 8)).astype(np.float32)
    assert np.allclose(a.q_values(x), b.q_values(x))
    q_on, q_tg = a.next_q(x)
    y = td_targets(np.ones(6, np.float32), np.zeros(6, bool), np.ones(6, int), 0.9, q_tg, q_on)
    assert y.shape == (6,) and y.dtype == np.float32

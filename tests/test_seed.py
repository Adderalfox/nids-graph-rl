import numpy as np

from nidsrl.utils.seed import seed_everything


def test_same_seed_same_numbers():
    a = seed_everything(123).random(5)
    b = seed_everything(123).random(5)
    assert np.array_equal(a, b)


def test_different_seed_different_numbers():
    a = seed_everything(1).random(5)
    b = seed_everything(2).random(5)
    assert not np.array_equal(a, b)


def test_global_numpy_state_is_seeded():
    seed_everything(7)
    a = np.random.rand(3)
    seed_everything(7)
    b = np.random.rand(3)
    assert np.array_equal(a, b)

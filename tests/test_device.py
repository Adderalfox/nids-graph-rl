import pytest

from nidsrl.utils.device import resolve_device


def test_cpu_always_works():
    assert resolve_device("cpu") == "cpu"


def test_auto_returns_valid_device():
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_invalid_preference_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")

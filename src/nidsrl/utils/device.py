"""Compute device resolution."""

from __future__ import annotations


def resolve_device(preference: str = "auto") -> str:
    """Map 'auto' | 'cpu' | 'cuda' to a usable device string.

    Falls back to CPU (with no error) when torch or CUDA is unavailable and the
    preference was 'auto'. An explicit 'cuda' request without CUDA raises.
    """
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto|cpu|cuda, got {preference!r}")
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if preference == "cuda":
            raise RuntimeError("device='cuda' requested but torch is not installed") from None
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return "cpu"

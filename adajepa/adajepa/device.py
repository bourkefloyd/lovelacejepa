"""Torch device selection (self-contained copy; no lab/ imports by design)."""

from __future__ import annotations

import torch


def torch_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = ["torch_device"]

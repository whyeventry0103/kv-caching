"""Determinism helpers for paired replay.

Bitwise self-replay should produce TV approximately zero. The analysis tau is a
fixed, interpretable TV level rather than an empirically tuned self-noise cutoff,
so it remains comparable across models, methods, budgets, and tasks.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from importlib import metadata


def set_determinism(seed: int) -> None:
    """Seed available RNGs without requiring Torch at import time."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextlib.contextmanager
def deterministic_kernels(enabled: bool = True) -> Iterator[None]:
    """Temporarily request deterministic Torch algorithms when Torch exists."""
    try:
        import torch
    except ImportError:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def log_versions() -> dict[str, str]:
    """Return required package versions, including explicit missing markers."""
    versions = {}
    for package in ("torch", "transformers", "kvpress"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions

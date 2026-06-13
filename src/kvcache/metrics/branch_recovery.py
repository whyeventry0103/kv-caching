"""Full-cache-fork recovery with a random-fork control."""

from __future__ import annotations


def controlled_recovery(fullcache_fork: float, random_fork: float) -> float:
    return fullcache_fork - random_fork

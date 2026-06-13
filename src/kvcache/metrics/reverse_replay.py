"""Replay a compressed free-run trace under the full cache."""

from __future__ import annotations

from collections.abc import Sequence

from kvcache.metrics.tdp import auc_tv


def reverse_replay_score(tv_profile: Sequence[float]) -> float:
    return auc_tv(tv_profile)

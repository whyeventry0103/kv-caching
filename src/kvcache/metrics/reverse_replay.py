"""Reverse replay: the compressed free-run trace replayed under the full cache.

CONTRACT (mandatory, not optional). Reverse replay is the symmetric counterpart to
forward (on-reference) teacher-forced replay and must be reported *alongside* it,
never instead of it. Forward replay measures readout on the reference (good-state)
trajectory; reverse replay measures it on the compressed model's own (worse-state)
trajectory. Together they **bracket** the on-policy readout from both sides and so
bound the off-policy bias in the decomposition residual (see
:mod:`kvcache.decomposition`). A profile statistic on the reference trajectory
alone cannot fix the covariate-shift problem (Ross & Bagnell, DAgger lineage);
reporting both replays is what converts "your residual is an artifact" into "we
bound the artifact."

This module currently exposes only the CPU summary of a stored TV profile. The
profile-producing engine is a GPU/M4 deliverable; it must feed
``decompose(..., reverse_readout_change=...)`` so the residual bracket is
populated rather than left as a single point estimate.
"""

from __future__ import annotations

from collections.abc import Sequence

from kvcache.metrics.tdp import auc_tv


def reverse_replay_score(tv_profile: Sequence[float]) -> float:
    """Summarize a reverse-replay TV profile (mean TV / normalized AUC)."""
    return auc_tv(tv_profile)

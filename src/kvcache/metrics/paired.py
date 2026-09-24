"""Paired accuracy contrasts between a compressed cell and its full-cache reference.

Amendment R.1 registers a paired, problem-clustered CI on the total change and
on the harmful and beneficial flip masses. Paired means per (problem, seed)
condition: the compressed and the full-cache run are matched on the same problem
and sampling seed, so a problem's difficulty cancels inside each difference.

Per condition, the net flip is ``correct_comp - correct_fc`` in {-1, 0, +1}, so
the mean net flip IS the total change on the matched population -- the two
quantities are one number, not two, and are reported as such. What the flip
decomposition adds is the split into lost (harmful: full cache right,
compressed wrong) and gained (beneficial: the reverse), each with a common
denominator of all matched conditions.

Resampling is by problem, carrying every seed of a problem together (the
project's standing convention, ``kvcache.stats.cluster_bootstrap_mean``).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any

from kvcache.stats import cluster_bootstrap_mean

__all__ = ["paired_flip_contrast", "trace_identity"]


def paired_flip_contrast(
    fc: Mapping[tuple[Hashable, Hashable], bool],
    comp: Mapping[tuple[Hashable, Hashable], bool],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Total change and flip masses over conditions present in BOTH arms.

    Keys are ``(problem_id, sampling_seed)``. Conditions in only one arm are
    counted and excluded, never imputed: a missing row is not a wrong answer.
    """
    shared = sorted(set(fc) & set(comp), key=lambda k: (str(k[0]), str(k[1])))
    if not shared:
        raise ValueError("no matched (problem, seed) conditions")
    net: dict[str, list[float]] = defaultdict(list)
    lost: dict[str, list[float]] = defaultdict(list)
    gained: dict[str, list[float]] = defaultdict(list)
    for key in shared:
        f, c = bool(fc[key]), bool(comp[key])
        pid = str(key[0])
        net[pid].append(float(c) - float(f))
        lost[pid].append(1.0 if (f and not c) else 0.0)
        gained[pid].append(1.0 if (c and not f) else 0.0)
    kw = {"n_resamples": n_resamples, "seed": seed}
    return {
        "total_change": cluster_bootstrap_mean(net, **kw),
        "harmful_flip_mass": cluster_bootstrap_mean(lost, **kw),
        "beneficial_flip_mass": cluster_bootstrap_mean(gained, **kw),
        "n_matched": len(shared),
        "n_only_fc": len(set(fc) - set(comp)),
        "n_only_comp": len(set(comp) - set(fc)),
        "note": "net flip mass == total_change on the matched population",
    }


def trace_identity(
    a: Mapping[Hashable, tuple[int, ...] | list[int]],
    b: Mapping[Hashable, tuple[int, ...] | list[int]],
) -> dict[str, Any]:
    """Token-for-token agreement of two generation sets on their shared keys.

    A determinism control, not an estimand (Amendment R.8.3): two runs that
    should be the same computation are compared trace by trace. For each
    differing trace the first differing token index is kept, because "differs
    at token 3" (a setup error) and "differs at token 9000" (accumulated
    floating-point drift) call for different responses.
    """
    shared = sorted(set(a) & set(b), key=repr)
    first_diff: dict[str, int] = {}
    for key in shared:
        x, y = list(a[key]), list(b[key])
        if x == y:
            continue
        idx = next(
            (i for i, (p, q) in enumerate(zip(x, y, strict=False)) if p != q),
            min(len(x), len(y)),
        )
        first_diff[repr(key)] = idx
    return {
        "n_shared": len(shared),
        "n_identical": len(shared) - len(first_diff),
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
        "first_difference": first_diff,
    }

"""Meeting #4 audit reads: paired residual contrasts and matched-key rankings.

Scientific rationale. The review of f69dce3 asked for two things the Meeting
#4 answer had stated from point estimates: (i) policy differences in the
residual with their paired uncertainty, not equality read off similar means;
(ii) accuracy and readout rankings on the SAME conditions the capture metrics
were measured on (25 problems at budget 256; the 21-problem / 69-trace
eligible set at 2048), before relating the two.

Residual per condition = free-run correctness minus readout correctness on
the conditions whose reference trace has an answer boundary (report_m4's
complete-case layer). It is a descriptive decomposition, not a measured causal
effect; the paired contrast between policies is reported with its CI.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from kvcache.coverage import equal_problem_bootstrap, paired_equal_problem

Key = tuple[str, int]


def per_condition_residual(
    free: Mapping[Key, bool], readout: Mapping[Key, bool]
) -> dict[Key, float]:
    """free − readout on the readout-defined conditions."""
    return {k: float(free[k]) - float(readout[k]) for k in readout if k in free}


def paired_policy_contrasts(
    free: Mapping[str, Mapping[Key, bool]],
    readout: Mapping[str, Mapping[Key, bool]],
    pairs: Collection[tuple[str, str]],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """For each (a, b): a − b in free-run, readout and residual, paired by key."""
    out: dict[str, Any] = {}
    for a, b in pairs:
        ra = per_condition_residual(free[a], readout[a])
        rb = per_condition_residual(free[b], readout[b])
        kw = {"n_resamples": n_resamples, "seed": seed}
        out[f"{a}-{b}"] = {
            "free": paired_equal_problem(
                {k: float(v) for k, v in free[a].items()},
                {k: float(v) for k, v in free[b].items()},
                **kw,
            ),
            "readout": paired_equal_problem(
                {k: float(v) for k, v in readout[a].items()},
                {k: float(v) for k, v in readout[b].items()},
                **kw,
            ),
            "residual": paired_equal_problem(ra, rb, **kw),
        }
    return out


def rankings_on_keys(
    free: Mapping[str, Mapping[Key, bool]],
    readout: Mapping[str, Mapping[Key, bool]],
    keys: Collection[Key],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per policy: free-run, readout and residual on exactly ``keys``."""
    keys = set(keys)
    out: dict[str, Any] = {"n_keys": len(keys), "per_policy": {}}
    for press in free:
        f = {k: float(free[press][k]) for k in keys if k in free[press]}
        r = {k: float(readout[press][k]) for k in keys if k in readout[press]}
        res = per_condition_residual(free[press], readout[press])
        res = {k: v for k, v in res.items() if k in keys}
        kw = {"n_resamples": n_resamples, "seed": seed}
        out["per_policy"][press] = {
            "n_free": len(f),
            "n_readout_defined": len(r),
            "free": equal_problem_bootstrap(f, **kw) if f else None,
            "readout": equal_problem_bootstrap(r, **kw) if r else None,
            "residual": equal_problem_bootstrap(res, **kw) if res else None,
        }
    return out

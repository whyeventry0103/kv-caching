"""Exact coverage and equal-problem estimation for Meeting #3 readers (R.14).

Two defects found in review of 4e38224 are fixed here once, for every reader:

1. **Completion was a row count.** Duplicated rows could stand in for missing
   conditions, a second budget could overwrite the first, and rows failing a
   control still counted. ``check_coverage`` compares the rows against the
   EXACT expected key set derived from the frozen design and reports missing,
   duplicated and unexpected keys; ``COMPLETE`` means none of the three.
   Coverage, control validity and estimand eligibility are separate fields: an
   undefined latent direction is an ineligible value, not a failed instrument.

2. **Problems were not equally weighted.** ``kvcache.stats.cluster_bootstrap_mean``
   resamples problems but averages the pooled observations, so a problem with
   four valid seeds outweighs one with a single valid seed. The registered
   estimand (R.6/R.9, R.3) averages seeds within problem, then problems.
   ``equal_problem_bootstrap`` computes exactly that estimator on the sample and
   on every bootstrap replicate. The historical helper is left untouched so no
   previously reported number moves.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product
from typing import Any

COMPLETE, PARTIAL, INVALID = "COMPLETE", "PARTIAL", "INVALID"


def expected_keys(*dims: Iterable[Hashable]) -> set[tuple]:
    """Cartesian product of the design's dimensions (problems, seeds, ...)."""
    return set(product(*[list(d) for d in dims]))


def check_coverage(
    rows: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], tuple],
    expected: set[tuple],
) -> dict[str, Any]:
    """Exact coverage of ``rows`` against ``expected`` keys.

    A row whose key cannot be formed (missing or mistyped fields) becomes an
    unexpected ``("<malformed>", i)`` key, so it makes the read INVALID instead
    of crashing the reader or being dropped.
    """
    got = []
    for i, r in enumerate(rows):
        try:
            got.append(key(r))
        except (KeyError, TypeError, ValueError):
            got.append(("<malformed>", i))
    counts = Counter(got)
    missing = sorted(expected - set(got), key=repr)
    dups = sorted((k for k, n in counts.items() if n > 1), key=repr)
    extra = sorted(set(got) - expected, key=repr)
    ok = not (missing or dups or extra)
    return {
        "status": COMPLETE if ok else (INVALID if (dups or extra) else PARTIAL),
        "n_expected": len(expected),
        "n_rows": len(rows),
        "n_unique_expected_present": len(expected & set(got)),
        "missing": [list(k) for k in missing],
        "duplicated": [list(k) for k in dups],
        "unexpected": [list(k) for k in extra],
    }


def equal_problem_mean(values: Mapping[tuple[str, int], float]) -> float:
    """Mean over problems of the mean over that problem's seeds."""
    per: dict[str, list[float]] = defaultdict(list)
    for (pid, _seed), v in values.items():
        per[pid].append(float(v))
    if not per:
        raise ValueError("no eligible values")
    return statistics.fmean(statistics.fmean(v) for v in per.values())


def equal_problem_bootstrap(
    values: Mapping[tuple[str, int], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Equal-problem mean with a problem-cluster percentile CI (same estimator
    on every replicate). ``values`` maps (problem, seed) -> eligible value."""
    per: dict[str, list[float]] = defaultdict(list)
    for (pid, _s), v in values.items():
        per[pid].append(float(v))
    if not per:
        return {"mean": None, "ci95": None, "n_problems": 0, "n_values": 0}
    means = {pid: statistics.fmean(v) for pid, v in per.items()}
    keys = sorted(means)
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(means[keys[rng.randrange(len(keys))]] for _ in keys)
        for _ in range(n_resamples)
    )
    return {
        "mean": statistics.fmean(means.values()),
        "ci95": [
            draws[int(0.025 * (n_resamples - 1))],
            draws[int(0.975 * (n_resamples - 1))],
        ],
        "n_problems": len(keys),
        "n_values": sum(len(v) for v in per.values()),
        "seeds_per_problem": {k: len(per[k]) for k in keys},
        "estimator": "equal-problem (seeds within problem, then problems)",
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_equal_problem(
    a: Mapping[tuple[str, int], float],
    b: Mapping[tuple[str, int], float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """a - b over (problem, seed) eligible in BOTH, then equal-problem averaging."""
    shared = set(a) & set(b)
    diffs = {k: float(a[k]) - float(b[k]) for k in shared}
    out = equal_problem_bootstrap(diffs, n_resamples=n_resamples, seed=seed)
    out["n_pairs"] = len(shared)
    out["excluded_only_in_a"] = len(set(a) - shared)
    out["excluded_only_in_b"] = len(set(b) - shared)
    if out["ci95"] is not None:
        lo, hi = out["ci95"]
        out["excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


FROZEN_SUBSET = "configs/meeting3/q2_subset_pids.json"


def design(config_path: str | Any, *, root: Any = None) -> dict[str, Any]:
    """The expected design of one Meeting #3 config, from FROZEN inputs only.

    Problems come from the committed ``configs/meeting3/q2_subset_pids.json``
    (verified identical to the dataset loader's 25-problem subset and to the
    Q.2 replay pids), never from a live dataset read, so the expected key set
    is reproducible anywhere. Seeds, policies, budgets, intervals and lags come
    from the config itself.
    """
    import json
    from pathlib import Path

    import yaml

    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((base / config_path).read_text(encoding="utf-8"))
    frozen = json.loads((base / FROZEN_SUBSET).read_text(encoding="utf-8"))
    n = int(cfg["subset"]["math500"])
    if n != int(frozen["subset"]) or len(frozen["pids"]) != n:
        raise ValueError(f"config subset {n} does not match the frozen subset file")
    return {
        "pids": list(frozen["pids"]),
        "seeds": [int(s) for s in cfg["seeds"]],
        "presses": list(cfg["presses"]),
        "budgets": [int(b) for b in cfg.get("budgets", []) if str(b) != "full"],
        "intervals": [int(i) for i in cfg.get("intervals", [])],
    }

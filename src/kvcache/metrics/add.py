"""Answer Distribution Distortion (ADD) metrics.

ADD is an **answer-level distribution divergence**: the total-variation distance
between the full-cache and compressed answer distributions. It is a *lower bound*
on behavioral change that only sees changes which propagate all the way to the
final answer. It is **not** a measure of behavioral change and must never be
described as one (a compression method can completely rewrite a reasoning trace
while ADD reads zero). Sub-answer behavior is the job of the trace-level
instruments (replay TV profile / TDP), which exist precisely because ADD is blind
below the answer.

Two contamination modes are made explicit here so they are never read as
reasoning change:

* **Termination mass.** When compression increases truncation/invalid output,
  probability moves into the ``OTHER`` bucket. That is a length effect, not a
  reasoning effect. :func:`add_decomposition` splits ADD into the termination-mass
  shift and the within-answered divergence.
* **Direction blindness.** ADD (like every divergence here) is reference-anchored
  and cannot tell damage from improvement. Every divergence must be reported with
  its *signed* outcome consequence: :func:`flip_mass` / :func:`aggregate_flip_mass`
  carry gained-vs-lost correct-answer mass. Divergence is "behavioral fidelity
  loss," never "damage"; correctness is the quality arbiter.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kvcache.grading import is_equiv
from kvcache.stats import exact_permutation_test

OTHER_ANSWER = "__OTHER__"
_INVALID_ANSWERS = {"", "none", "null", "nan", "invalid", "unknown"}


def coarsen_answer(
    answer: str | None,
    *,
    valid_support: set[str] | None = None,
    truncated: bool = False,
) -> str:
    """Map missing, truncated, or out-of-support answers to one frozen bucket."""
    if truncated or answer is None:
        return OTHER_ANSWER
    normalized = answer.strip()
    if normalized.lower() in _INVALID_ANSWERS:
        return OTHER_ANSWER
    if valid_support is not None and normalized not in valid_support:
        return OTHER_ANSWER
    return normalized


def answer_distribution(
    samples: Sequence[str | None],
    *,
    valid_support: set[str] | None = None,
) -> dict[str, float]:
    if not samples:
        raise ValueError("answer samples cannot be empty")
    counts = Counter(
        coarsen_answer(answer, valid_support=valid_support) for answer in samples
    )
    return {answer: count / len(samples) for answer, count in counts.items()}


def distribution_tv(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation between two answer distributions.

    Summed in SORTED key order, not set-iteration order. Floating-point addition
    is not associative, so an order that depends on which answers happened to be
    encountered first -- which in turn depends on the order the row files were
    globbed -- makes the result reproducible only to within a ULP. That is
    exactly what was observed: regenerating the ADD artifacts reproduced every
    aggregate bitwise but moved per-problem within_answered_tv by one ULP on 4 of
    100 rows. Sorting removes the dependence rather than documenting it, which
    matters because the byte-diff regenerability check and the nightly parity
    check both assume bitwise determinism, and a gate that flakes gets ignored.
    """
    return 0.5 * sum(
        abs(a.get(x, 0.0) - b.get(x, 0.0)) for x in sorted(a.keys() | b.keys())
    )


def add_tv(
    full: Sequence[str | None],
    compressed: Sequence[str | None],
    *,
    valid_support: set[str] | None = None,
) -> float:
    return distribution_tv(
        answer_distribution(full, valid_support=valid_support),
        answer_distribution(compressed, valid_support=valid_support),
    )


@dataclass(frozen=True)
class AddDecomposition:
    """ADD split into its termination-mass and within-answered channels."""

    add_tv: float
    termination_mass_full: float
    termination_mass_compressed: float
    termination_shift: float
    within_answered_tv: float | None


def add_decomposition(
    full: Sequence[str | None],
    compressed: Sequence[str | None],
    *,
    valid_support: set[str] | None = None,
) -> AddDecomposition:
    """Separate the termination-mass channel from the reasoning channel.

    ``add_tv`` is the full-support ADD. ``termination_shift`` is the signed change
    in ``OTHER`` mass (positive = compression terminated/invalidated more often).
    ``within_answered_tv`` is the TV between the *renormalized* answered-only
    distributions and is ``None`` when either condition produced no answered mass.
    The identity ``add_tv >= 0.5 * abs(termination_shift)`` always holds, so a
    large ``termination_shift`` flags a length-contaminated ADD whose reasoning
    content is carried by ``within_answered_tv`` instead.
    """
    full_dist = answer_distribution(full, valid_support=valid_support)
    comp_dist = answer_distribution(compressed, valid_support=valid_support)
    term_full = full_dist.get(OTHER_ANSWER, 0.0)
    term_comp = comp_dist.get(OTHER_ANSWER, 0.0)
    answered_full = {a: p for a, p in full_dist.items() if a != OTHER_ANSWER}
    answered_comp = {a: p for a, p in comp_dist.items() if a != OTHER_ANSWER}
    # Sorted summation, for the reason given in distribution_tv: these masses
    # are the renormalization denominators, so an order-dependent last bit here
    # propagates into every within-answered probability.
    mass_full = sum(answered_full[a] for a in sorted(answered_full))
    mass_comp = sum(answered_comp[a] for a in sorted(answered_comp))
    if mass_full > 0 and mass_comp > 0:
        renorm_full = {a: p / mass_full for a, p in answered_full.items()}
        renorm_comp = {a: p / mass_comp for a, p in answered_comp.items()}
        within = distribution_tv(renorm_full, renorm_comp)
    else:
        within = None
    return AddDecomposition(
        add_tv=distribution_tv(full_dist, comp_dist),
        termination_mass_full=term_full,
        termination_mass_compressed=term_comp,
        termination_shift=term_comp - term_full,
        within_answered_tv=within,
    )


def _correct_fraction(
    samples: Sequence[str | None],
    gold_raw: str,
    *,
    equiv: Callable[[str | None, str | None], bool],
) -> float:
    if not samples:
        raise ValueError("answer samples cannot be empty")
    return sum(equiv(gold_raw, answer) for answer in samples) / len(samples)


def flip_mass(
    full: Sequence[str | None],
    compressed: Sequence[str | None],
    gold_raw: str,
    *,
    equiv: Callable[[str | None, str | None], bool] = is_equiv,
) -> float:
    """Signed per-problem correct-mass change: ``P_comp(correct) - P_full(correct)``.

    Positive means compression *gained* correct mass on this problem (the >100%
    regime), negative means it *lost* correct mass. Carries the sign that ADD,
    being reference-anchored, cannot. ``equiv`` is the correctness authority and
    defaults to :func:`kvcache.grading.is_equiv`.
    """
    return _correct_fraction(compressed, gold_raw, equiv=equiv) - _correct_fraction(
        full, gold_raw, equiv=equiv
    )


def aggregate_flip_mass(per_problem: Sequence[float]) -> dict[str, float]:
    """Split signed per-problem flip mass into gained, lost, and net totals."""
    gained = sum(value for value in per_problem if value > 0)
    lost = sum(-value for value in per_problem if value < 0)
    return {"gained": gained, "lost": lost, "net": gained - lost}


def greedy_flip_rate(full: Sequence[str], compressed: Sequence[str]) -> float:
    """Greedy decoding uses paired answer flips, not sampled-inference ADD."""
    if len(full) != len(compressed):
        raise ValueError("paired greedy answers must have equal length")
    return (
        sum(a != b for a, b in zip(full, compressed, strict=True)) / len(full)
        if full
        else 0.0
    )


def excess_add_permutation(
    full: Sequence[str],
    compressed: Sequence[str],
    *,
    seed: int = 0,
) -> tuple[float, float]:
    return exact_permutation_test(full, compressed, add_tv, seed=seed)


def entropy_miller_madow(samples: Sequence[str]) -> float:
    distribution = answer_distribution(samples)
    entropy = -sum(p * math.log(p) for p in distribution.values() if p)
    return entropy + (len(distribution) - 1) / (2 * len(samples))


def truncation_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return the retained-row truncation rate for a result cell."""
    if not rows:
        return 0.0
    return sum(bool(row["truncated"]) for row in rows) / len(rows)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Codex pass@k estimator."""
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

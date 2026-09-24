"""Attention channel decomposition over shared KV pairs (Amendment Q.3).

The paper's central split is readout against trajectory, measured on an accuracy
scale. This module measures the same split one scale down, inside a single
attention operation, which is why it is worth running at all: an instrument that
merely produced another magnitude scalar would license nothing after Amendment
P.3 found that divergence magnitude does not predict accuracy loss within a
budget.

For a query at some decode step, with ``S`` the set of cache positions the
compressed model retained:

``M``  (mass_loss)  = 1 - sum of the FULL model's attention over ``S``.
       How much attention mass the full model wanted to place on positions the
       policy evicted. It depends only on the reference model and the eviction
       decision, so it is an **on-reference** quantity: computable without ever
       letting the compressed model run. This is the readout analogue.

``Delta`` (shape_tv) = TV between the compressed model's attention over ``S`` and
       the full model's attention **renormalised onto** ``S``. If compression
       only removed keys, and the compressed model's state were undamaged, these
       two would be equal and ``Delta`` would be zero, with the entire effect
       living in ``M``. Non-zero ``Delta`` means the compressed model's query was
       computed from an already-perturbed hidden state. This is the trajectory
       analogue.

The renormalisation is what makes the two distributions comparable at all: the
two softmaxes have different denominators, so the full model's raw attention over
``S`` is not a distribution. Dividing by its own mass on ``S`` is not a modelling
choice, it is the only way to compare shapes on a shared support.

**Self-control, and it is a gate rather than a diagnostic.** At the first prune
under teacher-forced replay both models have consumed identical token IDs, so the
query vectors are identical and ``Delta`` must be exactly ``0.0``. Amendment Q.3
makes a non-zero value a STOP condition: it means the harness introduced
divergence of its own, and no result from the cell is readable until it is
explained. This is the same standing as the exactly-zero pre-eviction TV control.

Everything here is pure Python over sequences so the CPU-only test gate keeps
meaning something; torch tensors are converted by the caller.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "mass_loss",
    "renormalised_restriction",
    "shape_tv",
    "collateral_fraction",
    "DEFAULT_COLLATERAL_TAU",
    "align_to_positions",
    "restricted_from_logits",
    "collateral_event",
]

# Fixed in Amendment Q.3 before the cell runs, so it cannot be tuned afterwards
# to make an ordering come out. A query head placing at least this much mass on
# an evicted position is treated as having wanted it.
DEFAULT_COLLATERAL_TAU = 0.01


def mass_loss(attention_full: Sequence[float], retained: Sequence[int]) -> float:
    """``M``: full-model attention mass that fell outside the retained set.

    Returns a value in [0, 1]. Indices in ``retained`` that fall outside the
    attention vector are ignored rather than raising, because a retained set may
    legitimately name positions beyond a truncated attention row.
    """
    n = len(attention_full)
    kept = sum(float(attention_full[j]) for j in set(retained) if 0 <= j < n)
    return max(0.0, 1.0 - kept)


def renormalised_restriction(
    attention_full: Sequence[float], retained: Sequence[int]
) -> list[float] | None:
    """``a_full|S``: the full model's attention restricted to ``S`` and rescaled.

    Returns ``None`` when the full model placed no mass at all on ``S``, which is
    the one case where the shape comparison is undefined rather than merely
    small. Callers record it as missing; they must not substitute a uniform
    distribution, which would read as "no distortion".
    """
    n = len(attention_full)
    idx = sorted(j for j in set(retained) if 0 <= j < n)
    total = sum(float(attention_full[j]) for j in idx)
    if total <= 0.0:
        return None
    return [float(attention_full[j]) / total for j in idx]


def shape_tv(
    attention_comp: Sequence[float],
    attention_full: Sequence[float],
    retained: Sequence[int],
) -> float | None:
    """``Delta``: total variation between the two distributions on ``S``.

    ``attention_comp`` is indexed over the retained positions in the same sorted
    order this module uses everywhere, so the caller does not have to guess the
    convention. Returns ``None`` when the restriction is undefined.
    """
    ref = renormalised_restriction(attention_full, retained)
    if ref is None:
        return None
    comp = [float(v) for v in attention_comp]
    if len(comp) != len(ref):
        raise ValueError(
            f"compressed attention has {len(comp)} entries but the retained set "
            f"has {len(ref)}; they must be aligned in sorted position order"
        )
    total = sum(comp)
    if total <= 0.0:
        return None
    comp = [v / total for v in comp]
    return 0.5 * sum(abs(a - b) for a, b in zip(comp, ref, strict=True))


def collateral_fraction(
    per_head_attention: Sequence[Sequence[float]],
    evicted: Sequence[int],
    *,
    tau: float = DEFAULT_COLLATERAL_TAU,
) -> float:
    """Fraction of evicted positions at least one query head in the group wanted.

    Under grouped-query attention the cache has fewer heads than the model has
    queries -- 4 against 28 on the primary model -- so ONE eviction decision is
    shared across the 7 query heads in a group. A position that only one of those
    heads cares about is evicted anyway when the group-aggregate score ranks it
    below the cut. This measures how often that happens.

    It is a cost that exists under GQA and does not exist under multi-head
    attention: eviction granularity is coarser than attention granularity. Every
    published eviction policy scores at KV-head granularity because that is where
    the cache lives, and none report what it costs.

    ``per_head_attention`` is the FULL model's attention, one row per query head
    in the group. Returns 0.0 when nothing was evicted, which is a true zero and
    not a missing value.
    """
    ev = [j for j in dict.fromkeys(evicted)]
    if not ev:
        return 0.0
    wanted = 0
    for j in ev:
        for row in per_head_attention:
            if 0 <= j < len(row) and float(row[j]) > tau:
                wanted += 1
                break
    return wanted / len(ev)


# ---------------------------------------------------------------------------
# Amendment R.4 reference implementations. The production capture computes on
# GPU tensors; these pure-Python versions are what it is parity-tested against.


def align_to_positions(
    values: Sequence[float], positions: Sequence[int], target: Sequence[int]
) -> list[float]:
    """Reorder ``values`` (aligned with ``positions``) onto ``target`` positions.

    kvpress keeps entries in SCORE order, not position order, and each KV head
    keeps a different set, so a compressed attention row is never safely
    assumed to be sorted. Identity is by absolute token position. A target
    position that is absent, or a duplicated source position, is a mapping
    error and raises: the permissive helpers above silently drop out-of-range
    indices, which is acceptable for old callers but not for production data.
    """
    if len(values) != len(positions):
        raise ValueError("values and positions must be the same length")
    index: dict[int, int] = {}
    for i, p in enumerate(positions):
        if int(p) in index:
            raise ValueError(f"duplicate position {p} in source mapping")
        index[int(p)] = i
    missing = [t for t in target if int(t) not in index]
    if missing:
        raise ValueError(f"target positions absent from source: {missing[:5]}")
    return [float(values[index[int(t)]]) for t in target]


def restricted_from_logits(
    logits: Sequence[float], keep: Sequence[int]
) -> tuple[float, list[float]]:
    """``(M, a_full|S)`` computed stably from logits by log-sum-exp.

    Computing from probabilities underflows for long contexts: a retained set
    can hold a real but tiny mass that rounds to zero, which would make the
    restriction undefined and tempt a caller to call Delta zero. Log-space
    avoids that. ``keep`` indexes into ``logits``.
    """
    import math

    lg = [float(x) for x in logits]
    if not keep:
        raise ValueError("empty retained set")
    lse_all = max(lg) + math.log(sum(math.exp(x - max(lg)) for x in lg))
    sub = [lg[i] for i in keep]
    m = max(sub)
    lse_s = m + math.log(sum(math.exp(x - m) for x in sub))
    mass_loss_value = -math.expm1(lse_s - lse_all)  # 1 - exp(lse_S - lse_all)
    return max(0.0, mass_loss_value), [math.exp(x - lse_s) for x in sub]


def collateral_event(
    per_head_attention: Sequence[Sequence[float]],
    positions: Sequence[int],
    evicted: Sequence[int],
    *,
    budget: int,
    tau: float = DEFAULT_COLLATERAL_TAU,
    protected: Sequence[int] = (),
) -> dict[str, float | int | None]:
    """Registered GQA collateral for one (layer, KV group, event) -- Amendment R.4.

    A position counts as collateral when BOTH hold: (a) at least one query head
    of the group placed more than ``tau`` of its full-reference attention on it,
    and (b) the group's MEAN full-reference attention ranks it outside the top
    ``budget`` candidates (ties toward the lower position). That second
    condition is what makes this a GQA cost -- wanted by an individual head but
    not by the group average -- and it is exactly what ``collateral_fraction``
    omits. That older statistic is returned here as ``wanted_any_head`` and is
    never labelled collateral.

    ``per_head_attention`` rows are aligned with ``positions`` (absolute token
    ids of every candidate at the event). Protected positions are excluded from
    the denominator. With nothing eligible evicted the fractions are ``None``
    (unavailable), never 0.
    """
    n_heads = len(per_head_attention)
    if n_heads == 0:
        raise ValueError("no query heads")
    for row in per_head_attention:
        if len(row) != len(positions):
            raise ValueError("attention rows must align with positions")
    col = {int(p): i for i, p in enumerate(positions)}
    if len(col) != len(positions):
        raise ValueError("duplicate candidate positions")
    group_mean = [
        sum(float(r[i]) for r in per_head_attention) / n_heads
        for i in range(len(positions))
    ]
    order = sorted(range(len(positions)), key=lambda i: (-group_mean[i], positions[i]))
    top = {int(positions[i]) for i in order[: int(budget)]}
    prot = {int(p) for p in protected}
    eligible = [int(j) for j in dict.fromkeys(evicted) if int(j) not in prot]
    unknown = [j for j in eligible if j not in col]
    if unknown:
        raise ValueError(f"evicted positions not among candidates: {unknown[:5]}")
    wanted = [
        j for j in eligible if any(float(r[col[j]]) > tau for r in per_head_attention)
    ]
    collateral = [j for j in wanted if j not in top]
    n = len(eligible)
    return {
        "n_evicted_eligible": n,
        "n_wanted_any_head": len(wanted),
        "n_collateral": len(collateral),
        "wanted_any_head": (len(wanted) / n) if n else None,
        "collateral": (len(collateral) / n) if n else None,
        "tau": float(tau),
        "budget_cut": int(budget),
    }

"""Exact information through a finite readout channel (Meeting 5, M5-C;
prepared 2026-10-14, before any information-probe outcome exists).

Scientific rationale. Raw cache-to-cache mutual information rewards any retained
variability, and for a deterministic benchmark I(Y; cache | question) is zero.
The Meeting 5 review therefore replaces it with a controlled channel: two
independent, uniformly randomized four-valued facts, a target S and a nuisance
N, pass through the decode cache; a short readout that cannot evict again asks
for one of them; the native model distribution over the four single-token
answer symbols plus OTHER (all remaining vocabulary mass) is the channel
output Z. For one context u, every one of the 16 (S, N) combinations is run, so

    p(z | s, u) = 1/4 sum_n q(z | s, n, u)          (nuisance marginalized)
    pbar(z | u) = 1/4 sum_s p(z | s, u)
    I(S; Z | u) = 1/4 sum_s sum_z p(z | s, u) log2 p(z | s, u) / pbar(z | u)

is EXACT for that context -- no output sampling. It is information through the
declared answer channel, bounded by log2(4) = 2 bits; it is NOT the information
in the whole cache, and every output label says so (``ENDPOINT_LABEL``).

Information is not correct use: a fixed wrong permutation of the labels
transmits 2 bits with correct probability 0. So every channel is reported with
the expected correct-symbol probability, OTHER mass, native greedy accuracy and
proper log loss over the UNRENORMALIZED answer probabilities (format failures
are not normalized away). Log loss is computed from log-probabilities; a
mathematical zero gives an infinite loss, never a clipped one.

Grid discipline: a context must contain each of the 16 (S, N) combinations
exactly once. A missing combination makes the context PARTIAL (the prior is
never re-weighted to fit what exists); a duplicate, an out-of-range value or an
invalid distribution makes it INVALID.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

K = 4  # values per variable; answer symbols 0..K-1, OTHER = index K
OTHER = K
LOG2 = math.log(2.0)
# Declared 2026-10-14, before any outcome: the symbol probabilities of one
# readout, gathered from a log-softmax over the vocabulary, may exceed 1 by at
# most this much (float rounding). Beyond it the row is INVALID; within it the
# OTHER mass is recorded as a numerical zero and flagged, not as a fact.
SUM_TOL = 1e-6
ENDPOINT_LABEL = "output-channel MI (bits) through the declared answer channel"
NOT_LABEL = "not total cache MI"

FINAL = "FINAL"
PARTIAL = "PARTIAL"
INVALID = "INVALID"


@dataclass(frozen=True)
class Readout:
    """One readout of one (S, N) variant of one context.

    ``logprobs``: natural-log probabilities of the K answer symbols, in value
    order, as gathered from the native full-vocabulary log-softmax.
    ``greedy``: the value whose symbol is the native top-1 token, or None when
    the top-1 token is not an answer symbol.
    """

    s: int
    n: int
    logprobs: tuple[float, ...]
    greedy: int | None


def channel_row(symbol_logprobs: Sequence[float]) -> tuple[list[float], bool]:
    """Log-probabilities of the K + 1 categories, and an OTHER-at-resolution flag.

    The flag is set when the symbol mass rounds to 1 or above, so OTHER is
    recorded as 0. In an analytic fixture that zero is exact; in a model
    readout it is OTHER mass below float resolution. The flag cannot tell the
    two apart -- the data source does -- so it is counted, never interpreted.

    OTHER = 1 - sum(symbols), computed as log(-expm1(logsumexp)) so a small
    OTHER mass is not lost to cancellation. Raises ValueError on a non-finite
    or impossible input.
    """
    if len(symbol_logprobs) != K:
        raise ValueError(f"expected {K} symbol log-probabilities")
    lp = [float(x) for x in symbol_logprobs]
    if any(math.isnan(x) or x > 0 or x == math.inf for x in lp):
        raise ValueError(f"invalid log-probability in {lp}")
    top = max(lp)
    if top == -math.inf:
        return lp + [0.0], False  # all symbols exactly impossible: all OTHER
    lse = top + math.log(sum(math.exp(x - top) for x in lp))
    if lse > math.log1p(SUM_TOL):
        raise ValueError(f"symbol mass {math.exp(lse)} exceeds 1 + {SUM_TOL}")
    if lse >= 0.0:
        return lp + [-math.inf], True
    return lp + [math.log(-math.expm1(lse))], False


def grid_status(readouts: Iterable[Readout]) -> tuple[str, list[str]]:
    """FINAL iff every (S, N) in K x K appears exactly once with a valid row."""
    seen: dict[tuple[int, int], int] = {}
    problems: list[str] = []
    status = FINAL
    for r in readouts:
        if not (0 <= r.s < K and 0 <= r.n < K):
            problems.append(f"value out of range: s={r.s} n={r.n}")
            status = INVALID
            continue
        seen[(r.s, r.n)] = seen.get((r.s, r.n), 0) + 1
        if r.greedy is not None and not 0 <= r.greedy < K:
            problems.append(f"greedy symbol out of range at s={r.s} n={r.n}")
            status = INVALID
        try:
            channel_row(r.logprobs)
        except ValueError as exc:
            problems.append(f"s={r.s} n={r.n}: {exc}")
            status = INVALID
    dups = sorted(k for k, c in seen.items() if c > 1)
    if dups:
        problems.append(f"duplicate variants {dups}")
        status = INVALID
    missing = [(s, n) for s in range(K) for n in range(K) if (s, n) not in seen]
    if missing and status != INVALID:
        status = PARTIAL
    if missing:
        problems.append(f"missing variants {missing}")
    return status, problems


def _rows(readouts: Iterable[Readout]) -> dict[tuple[int, int], Readout]:
    rows = list(readouts)
    status, problems = grid_status(rows)
    if status != FINAL:
        raise ValueError(f"{status} grid: {problems}")
    return {(r.s, r.n): r for r in rows}


def mutual_information_bits(readouts: Iterable[Readout], target: str) -> float:
    """I(target; Z | u) in bits, the other variable marginalized uniformly.

    ``target`` is "s" or "n": which input variable the information is about.
    The same readouts can be scored against either (a channel that reports N
    carries 0 bits about S). Uses 0 log 0 = 0, valid because pbar(z) >= p(z|v)/K.
    """
    rows = _rows(readouts)
    if target not in ("s", "n"):
        raise ValueError("target must be 's' or 'n'")
    probs = {
        key: [math.exp(x) for x in channel_row(r.logprobs)[0]]
        for key, r in rows.items()
    }
    cond = []
    for v in range(K):
        keys = [(v, o) if target == "s" else (o, v) for o in range(K)]
        cond.append([sum(probs[k][z] for k in keys) / K for z in range(K + 1)])
    marg = [sum(cond[v][z] for v in range(K)) / K for z in range(K + 1)]
    mi = 0.0
    for v in range(K):
        for z in range(K + 1):
            p = cond[v][z]
            if p > 0.0:
                mi += p * math.log2(p / marg[z])
    mi /= K
    if mi < -1e-12:  # MI >= 0 exactly; a larger negative is a bug, not rounding
        raise RuntimeError(f"negative mutual information {mi}")
    return max(0.0, mi)


def channel_summary(readouts: Iterable[Readout], query: str) -> dict[str, float]:
    """All declared quantities for one context and one query ("s" or "n").

    The correct symbol for a readout is the queried variable's value. Log loss
    is the mean of -log2 q(correct) over the 16 variants: infinite if any
    correct probability is exactly zero.
    """
    rows = _rows(readouts)
    if query not in ("s", "n"):
        raise ValueError("query must be 's' or 'n'")
    other = "n" if query == "s" else "s"
    correct_p = other_p = loss = greedy = 0.0
    at_resolution = 0
    for (s, n), r in rows.items():
        row, flag = channel_row(r.logprobs)
        at_resolution += int(flag)
        want = s if query == "s" else n
        correct_p += math.exp(row[want])
        other_p += math.exp(row[OTHER])
        loss += -row[want] / LOG2 if row[want] > -math.inf else math.inf
        greedy += float(r.greedy == want)
    m = float(K * K)
    return {
        "mi_bits": mutual_information_bits(rows.values(), query),
        "mi_other_variable_bits": mutual_information_bits(rows.values(), other),
        "expected_correct_prob": correct_p / m,
        "other_prob": other_p / m,
        "log_loss_bits": loss / m,
        "reader_score_bits": math.log2(K) - loss / m,
        "greedy_accuracy": greedy / m,
        "other_at_resolution_rows": float(at_resolution),
    }


# --- analytic reference channels (review section 10.3) ---------------------


def _log(p: float) -> float:
    return math.log(p) if p > 0 else -math.inf


def reference_channel(
    kind: str, *, query: str = "s", perm: Sequence[int] = (1, 2, 3, 0)
) -> list[Readout]:
    """The mandatory reference cases as complete 16-variant grids.

    identity: the queried value's symbol with probability 1 (2 bits, correct 1);
    uniform: 1/K on every symbol regardless of input (0 bits, correct 1/K);
    permutation: symbol perm[value] with probability 1 (2 bits, correct 0 for a
    derangement); always_other: every symbol probability 0 (0 bits, correct 0);
    copy_other: reports the OTHER variable perfectly (0 bits about the query).
    """
    out = []
    for s in range(K):
        for n in range(K):
            v, o = (s, n) if query == "s" else (n, s)
            if kind == "identity":
                hot = v
            elif kind == "permutation":
                hot = perm[v]
            elif kind == "copy_other":
                hot = o
            elif kind == "uniform":
                lp = tuple([_log(1.0 / K)] * K)
                out.append(Readout(s, n, lp, None))
                continue
            elif kind == "always_other":
                out.append(Readout(s, n, tuple([-math.inf] * K), None))
                continue
            else:
                raise ValueError(f"unknown reference channel {kind!r}")
            lp = tuple(0.0 if z == hot else -math.inf for z in range(K))
            out.append(Readout(s, n, lp, hot))
    return out


def reference_table() -> list[Mapping[str, float | str]]:
    """The section 10.3 table, recomputed from the implementation."""
    rows: list[Mapping[str, float | str]] = []
    for kind in ("identity", "uniform", "permutation", "always_other"):
        summ = channel_summary(reference_channel(kind), "s")
        rows.append(
            {
                "channel": kind,
                "mi_bits": summ["mi_bits"],
                "expected_correct_prob": summ["expected_correct_prob"],
            }
        )
    return rows


# --- the reader: rows -> FINAL / PARTIAL / INVALID read ------------------------

ROW_FIELDS = ("policy", "query", "context_id", "eviction_seed", "s", "n")
METRICS = (
    "mi_bits",
    "mi_other_variable_bits",
    "expected_correct_prob",
    "other_prob",
    "log_loss_bits",
    "reader_score_bits",
    "greedy_accuracy",
)


def readout_from_row(row: Mapping[str, Any]) -> Readout:
    """Parse one stored readout row (raises ValueError when malformed)."""
    for f in (*ROW_FIELDS, "symbol_logprobs", "greedy"):
        if f not in row:
            raise ValueError(f"row lacks {f!r}")
    lp = row["symbol_logprobs"]
    if not isinstance(lp, list | tuple):
        raise ValueError("symbol_logprobs is not a list")
    g = row["greedy"]
    return Readout(
        int(row["s"]),
        int(row["n"]),
        tuple(float(x) for x in lp),
        None if g is None else int(g),
    )


def read_information_probe(
    rows: Iterable[Mapping[str, Any]],
    design: Mapping[str, Any],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate every row against the frozen design and, only if FINAL, label.

    ``design``: ``policies``, ``reference_policy``, ``contexts`` (ids),
    ``eviction_seeds`` and ``queries`` -- derived from the frozen manifest,
    never from the rows. Every (policy, query, context, eviction seed) cell must
    be a complete 16-variant grid. Per policy and query, metrics are averaged
    over eviction seeds within a context, then over contexts (contexts are the
    sampling unit; seeds are not independent evidence). Paired contrasts
    against ``reference_policy`` resample contexts (2000 resamples, seed 0).
    A PARTIAL read reports coverage only; INVALID fails closed.
    """
    from kvcache.stats import cluster_bootstrap_mean

    policies = [str(p) for p in design["policies"]]
    contexts = [str(c) for c in design["contexts"]]
    seeds = [int(s) for s in design["eviction_seeds"]]
    queries = [str(q) for q in design["queries"]]
    ref = str(design["reference_policy"])
    if ref not in policies or not set(queries) <= {"s", "n"}:
        raise ValueError("design: reference policy or queries malformed")
    expected = {
        (p, q, c, e) for p in policies for q in queries for c in contexts for e in seeds
    }
    cells: dict[tuple[str, str, str, int], list[Readout]] = {}
    problems: list[str] = []
    status = FINAL
    for i, row in enumerate(rows):
        try:
            key = (
                str(row["policy"]),
                str(row["query"]),
                str(row["context_id"]),
                int(row["eviction_seed"]),
            )
            r = readout_from_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"row {i}: {exc}")
            status = INVALID
            continue
        if key not in expected:
            problems.append(f"row {i}: cell {key} is not in the design")
            status = INVALID
            continue
        cells.setdefault(key, []).append(r)
    summaries: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for key in sorted(expected):
        if key not in cells:
            continue
        st, why = grid_status(cells[key])
        if st == INVALID:
            status = INVALID
            problems.append(f"{key}: {why}")
        elif st == PARTIAL:
            problems.append(f"{key}: {why}")
        else:
            summaries[key] = channel_summary(cells[key], key[1])
    missing = sorted(expected - set(summaries))
    if missing and status == FINAL:
        status = PARTIAL
    report: dict[str, Any] = {
        "status": status,
        "endpoint": ENDPOINT_LABEL,
        "not": NOT_LABEL,
        "n_expected_cells": len(expected),
        "n_complete_cells": len(summaries),
        "missing_cells": [list(k) for k in missing[:20]],
        "problems": problems[:50],
    }
    if status != FINAL:
        return report  # no label, no number for an incomplete or invalid read

    def by_context(p: str, q: str, m: str) -> dict[str, float]:
        return {
            c: sum(summaries[(p, q, c, e)][m] for e in seeds) / len(seeds)
            for c in contexts
        }

    table: dict[str, dict[str, dict[str, Any]]] = {}
    for p in policies:
        for q in queries:
            entry: dict[str, Any] = {}
            for m in METRICS:
                vals = by_context(p, q, m)
                if all(math.isfinite(v) for v in vals.values()):
                    bs = cluster_bootstrap_mean(
                        {c: [v] for c, v in vals.items()},
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                    entry[m] = {"mean": bs["mean"], "ci95": bs["ci95"]}
                else:
                    entry[m] = {"mean": None, "ci95": None, "non_finite": True}
            if p != ref:
                a, b = by_context(p, q, "mi_bits"), by_context(ref, q, "mi_bits")
                bs = cluster_bootstrap_mean(
                    {c: [a[c] - b[c]] for c in contexts},
                    n_resamples=n_resamples,
                    seed=seed,
                )
                entry[f"mi_bits_minus_{ref}"] = {"mean": bs["mean"], "ci95": bs["ci95"]}
            table.setdefault(p, {})["query_" + q] = entry
    report.update(
        policies=table,
        n_contexts=len(contexts),
        eviction_seeds=seeds,
        bootstrap={"unit": "context", "n_resamples": n_resamples, "seed": seed},
    )
    return report

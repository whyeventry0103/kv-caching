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
                # all mass on tied symbols: native argmax picks the first
                out.append(Readout(s, n, lp, 0))
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
INT_FIELDS = ("eviction_seed", "s", "n")
ROW_SCHEMA = "info-readout-v1"
REFERENCE, PRODUCTION = "reference", "production"
# Production rows must carry every control the runner computes, each passed
# (review of 6b409001, R3): two independent full-cache arms agree exactly; two
# independent clones of the checkpoint cache read out identically; and the
# readout itself evicted nothing.
REQUIRED_CONTROLS = ("full_full_identity", "clone_agreement", "no_readout_eviction")
PROVENANCE_FIELDS = (
    "implementation_id",
    "model_revision",
    "tokenizer_revision",
    "answer_alphabet_sha256",
    "task_layout_sha256",
    "design_sha256",
    # the actual selected inputs (filler passages, blocks, rendered token ids;
    # review of 5616630, R2), not only the config that chose them
    "input_manifest_sha256",
)
FACTS = ("s", "n")
GREEDY_TOL = 1e-9  # log-space tolerance for "this symbol is the argmax"
METRICS = (
    "mi_bits",
    "mi_other_variable_bits",
    "expected_correct_prob",
    "other_prob",
    "log_loss_bits",
    "reader_score_bits",
    "greedy_accuracy",
)


def _strict_int(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name} is not an integer: {v!r}")
    return v


def greedy_problem(r: Readout, top1_logprob: float | None) -> str | None:
    """Is the declared native greedy symbol consistent with the stored row?

    A declared symbol must be the argmax among the answer symbols (and, when
    the native top-1 log-probability is stored, equal it). A declared non-symbol
    top-1 (None) needs OTHER mass at least the largest symbol probability --
    a single non-symbol token cannot beat a symbol it is smaller than.
    """
    row, _ = channel_row(r.logprobs)
    best = max(row[:K])
    if r.greedy is not None:
        if row[r.greedy] < best - GREEDY_TOL or row[r.greedy] == -math.inf:
            return f"greedy symbol {r.greedy} is not the argmax of the stored row"
        if top1_logprob is not None and abs(top1_logprob - row[r.greedy]) > 1e-6:
            return "greedy symbol's log-probability differs from the stored top-1"
        return None
    if best > row[OTHER] + GREEDY_TOL:
        return "non-symbol greedy token although a symbol outweighs all OTHER mass"
    if top1_logprob is not None and not (best - GREEDY_TOL <= top1_logprob):
        return "stored top-1 is below the best symbol although greedy is OTHER"
    return None


def readout_from_row(
    row: Mapping[str, Any],
    *,
    mode: str,
    expected_provenance: Mapping[str, Any] | None = None,
) -> Readout:
    """Parse and validate one stored readout row (ValueError when malformed)."""
    for f in (*ROW_FIELDS, "symbol_logprobs", "greedy"):
        if f not in row:
            raise ValueError(f"row lacks {f!r}")
    for f in INT_FIELDS:
        _strict_int(row[f], f)
    lp = row["symbol_logprobs"]
    if not isinstance(lp, list) or not all(
        isinstance(x, int | float) and not isinstance(x, bool) for x in lp
    ):
        raise ValueError("symbol_logprobs is not a list of numbers")
    g = row["greedy"]
    if g is not None:
        _strict_int(g, "greedy")
    r = Readout(row["s"], row["n"], tuple(float(x) for x in lp), g)
    channel_row(r.logprobs)  # validates the distribution itself
    top1 = row.get("top1_logprob")
    if mode == PRODUCTION:
        if row.get("schema") != ROW_SCHEMA:
            raise ValueError(f"schema {row.get('schema')!r} is not {ROW_SCHEMA}")
        prov = row.get("provenance")
        if not isinstance(prov, dict):
            raise ValueError("production row without provenance")
        for f in PROVENANCE_FIELDS:
            if prov.get(f) != (expected_provenance or {}).get(f):
                raise ValueError(f"provenance {f} differs from the design's")
        controls = row.get("controls")
        if not isinstance(controls, dict):
            raise ValueError("production row without controls")
        for c in REQUIRED_CONTROLS:
            if controls.get(c) is not True:
                raise ValueError(f"control {c} did not pass: {controls.get(c)!r}")
        if not (isinstance(top1, float | int) and math.isfinite(top1) and top1 <= 0):
            raise ValueError(f"top1_logprob missing or invalid: {top1!r}")
    elif mode != REFERENCE:
        raise ValueError(f"unknown design mode {mode!r}")
    elif "provenance" in row or "controls" in row:
        raise ValueError("reference-mode fixtures carry no provenance or controls")
    why = greedy_problem(r, float(top1) if top1 is not None else None)
    if why:
        raise ValueError(why)
    return r


def _finite_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def measurement_schedule(
    design: Mapping[str, Any], contexts: Sequence[str], queries: Sequence[str]
) -> dict[str, Any]:
    """The frozen schedule of the calibration measurements (review of 5616630,
    R1): which query carries the end-to-end control, and each context's fact
    spans. A production design must declare it; ValueError when malformed."""
    sched = design.get("measurement_schedule")
    if not isinstance(sched, Mapping):
        raise ValueError("a production design must declare measurement_schedule")
    e2e = sched.get("e2e_query")
    if e2e not in queries:
        raise ValueError(f"measurement_schedule e2e_query {e2e!r} is not a query")
    spans = sched.get("fact_spans")
    if not isinstance(spans, Mapping) or set(spans) != set(contexts):
        raise ValueError("measurement_schedule fact_spans must cover every context")
    for c, sp in spans.items():
        if not isinstance(sp, Mapping) or set(sp) != set(FACTS):
            raise ValueError(f"fact_spans of {c} must name exactly {FACTS}")
        for f, ab in sp.items():
            if not (
                isinstance(ab, list)
                and len(ab) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in ab)
                and 0 <= ab[0] < ab[1]
            ):
                raise ValueError(f"fact_spans {c}.{f} malformed: {ab!r}")
    return {"e2e_query": e2e, "fact_spans": spans}


def measurement_problem(
    row: Mapping[str, Any], schedule: Mapping[str, Any], reference_policy: str
) -> str | None:
    """Why a production row's calibration measurements are unusable, or None.

    Validity, not acceptance: every measurement the calibration consumes must
    be present exactly where the frozen schedule says it is measured, typed and
    in range, BEFORE any aggregation (a missing value must not drop out of a
    mean, a NaN must not vanish inside max(), a negative fraction must not
    satisfy an upper bound). Review of 5616630, R1.
    """
    ref = str(row["policy"]) == reference_policy
    prunes = row.get("n_prunes")
    if isinstance(prunes, bool) or not isinstance(prunes, int) or prunes < 0:
        return f"n_prunes is not a non-negative integer: {prunes!r}"
    if ref and prunes != 0:
        return "the reference (full-cache) arm recorded prunes"
    if row.get("fact_spans") != schedule["fact_spans"][str(row["context_id"])]:
        return "fact_spans differ from the frozen schedule"
    kept = row.get("retained_fraction")
    if not isinstance(kept, Mapping) or set(kept) != set(FACTS):
        return f"retained_fraction must name exactly {FACTS}"
    for f in FACTS:
        v = kept[f]
        if ref:
            if v is not None:
                return "the reference arm carries a retained fraction"
        elif not (_finite_number(v) and 0.0 <= v <= 1.0):
            return f"retained_fraction.{f} missing or outside [0, 1]: {v!r}"
    e2e = row.get("e2e_max_abs_diff")
    if str(row["query"]) == schedule["e2e_query"]:
        if not (_finite_number(e2e) and e2e >= 0.0):
            return f"end-to-end control missing or invalid: {e2e!r}"
    elif e2e is not None:
        return "end-to-end value on a query the schedule does not measure"
    return None


def _sha256_json(obj: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def design_bindings(
    design: Mapping[str, Any],
    manifest: Any,
    criteria_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify what a production design binds; return the dependency record.

    Review of 5616630: the stored input manifest must hash to the design's
    ``input_manifest_sha256`` (R2), and the calibration criteria are the
    design's own, matching their fingerprint; a criteria file named on the
    command line must be the SAME criteria, never a substitute (R1.4).
    ValueError on any mismatch (the read is then INVALID)."""
    prov = design.get("expected_provenance") or {}
    want = prov.get("input_manifest_sha256")
    if not isinstance(want, str) or _sha256_json(manifest) != want:
        raise ValueError("input manifest does not match the design's hash")
    crit = design.get("calibration_criteria")
    crit_sha = design.get("calibration_criteria_sha256")
    if not isinstance(crit, Mapping) or _sha256_json(crit) != crit_sha:
        raise ValueError("design calibration criteria do not match their fingerprint")
    if criteria_override is not None and _sha256_json(criteria_override) != crit_sha:
        raise ValueError(
            "--calibration-criteria differs from the criteria frozen in the design"
        )
    return {
        "input_manifest_sha256": want,
        "calibration_criteria_sha256": crit_sha,
        "criteria_source": "design (file matched)" if criteria_override else "design",
    }


def read_information_probe(
    rows: Iterable[Mapping[str, Any]],
    design: Mapping[str, Any],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate every row against the frozen design and, only if FINAL, label.

    ``design``: ``mode`` (``reference`` for analytic fixtures, ``production``
    for model readouts, which must carry the versioned schema, the
    ``expected_provenance`` and passed controls), ``policies``,
    ``reference_policy``, ``contexts``, ``eviction_seeds``, ``queries`` and
    optional ``paired_contrasts`` ([[a, b], ...]) -- from the frozen manifest,
    never from the rows. A production design also declares the
    ``measurement_schedule`` its calibration measurements are validated
    against (``measurement_problem``). Every (policy, query, context, eviction
    seed) cell must be a complete 16-variant grid. Per policy and query,
    metrics are averaged over eviction seeds within a context, then over
    contexts. Contrasts are
    computed from PAIRED context-level values and resample contexts (2000,
    seed 0); intervals are never subtracted. PARTIAL reports coverage only;
    INVALID fails closed.
    """
    from kvcache.stats import cluster_bootstrap_mean

    mode = str(design.get("mode", ""))
    expected_prov = design.get("expected_provenance")
    if mode not in (REFERENCE, PRODUCTION):
        raise ValueError(f"design mode must be {REFERENCE!r} or {PRODUCTION!r}")
    if mode == PRODUCTION and not isinstance(expected_prov, dict):
        raise ValueError("a production design must declare expected_provenance")
    policies = [str(p) for p in design["policies"]]
    contexts = [str(c) for c in design["contexts"]]
    seeds = [_strict_int(s, "design eviction seed") for s in design["eviction_seeds"]]
    queries = [str(q) for q in design["queries"]]
    ref = str(design["reference_policy"])
    pairs = [(str(a), str(b)) for a, b in design.get("paired_contrasts", [])]
    if ref not in policies or not set(queries) <= {"s", "n"}:
        raise ValueError("design: reference policy or queries malformed")
    if any(a not in policies or b not in policies for a, b in pairs):
        raise ValueError("design: a paired contrast names an unknown policy")
    expected = {
        (p, q, c, e) for p in policies for q in queries for c in contexts for e in seeds
    }
    schedule = (
        measurement_schedule(design, contexts, queries) if mode == PRODUCTION else None
    )
    cells: dict[tuple[str, str, str, int], list[Readout]] = {}
    problems: list[str] = []
    status = FINAL
    for i, row in enumerate(rows):
        try:
            if not isinstance(row, Mapping):
                raise ValueError("row is not an object")
            key = (
                str(row["policy"]),
                str(row["query"]),
                str(row["context_id"]),
                _strict_int(row["eviction_seed"], "eviction_seed"),
            )
            r = readout_from_row(row, mode=mode, expected_provenance=expected_prov)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"row {i}: {exc}")
            status = INVALID
            continue
        if key not in expected:
            problems.append(f"row {i}: cell {key} is not in the design")
            status = INVALID
            continue
        why = measurement_problem(row, schedule, ref) if schedule else None
        if why:
            problems.append(f"row {i}: {why}")
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
        "mode": mode,
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

    def paired(a: str, b: str, q: str) -> dict[str, Any]:
        va, vb = by_context(a, q, "mi_bits"), by_context(b, q, "mi_bits")
        bs = cluster_bootstrap_mean(
            {c: [va[c] - vb[c]] for c in contexts},
            n_resamples=n_resamples,
            seed=seed,
        )
        return {"mean": bs["mean"], "ci95": bs["ci95"], "n_contexts": len(contexts)}

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
                entry[f"mi_bits_minus_{ref}"] = paired(p, ref, q)
            table.setdefault(p, {})["query_" + q] = entry
    report.update(
        policies=table,
        paired_contrasts={
            f"{a}-{b}": {f"query_{q}": paired(a, b, q) for q in queries}
            for a, b in pairs
        },
        n_contexts=len(contexts),
        eviction_seeds=seeds,
        bootstrap={"unit": "context", "n_resamples": n_resamples, "seed": seed},
    )
    return report

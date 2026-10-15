"""Realized compression of free-run grid cells (2026-10-15; corrected
2026-10-16 after the review of 5616630, R4).

Scientific rationale. The framing rules require every absolute budget to be
reported with its realized compression ratio. What each grid row records in
``compression_ratio`` is a NOMINAL-BUDGET ratio, min(1, B / (prompt +
generated tokens)): a budget-to-sequence-length ratio. It is not live cache
occupancy, K/V bytes, or a count of eviction events, and the labels say so.

* Ratio 1 means the sequence never exceeded B, so nothing could be pruned.
  Ratio < 1 does NOT prove a prune happened (pruning is periodic and the rows
  keep no prune count), so ``n_ratio_one`` is a LOWER bound on the
  never-compressed conditions, never the never-compressed count itself.
* Where the frozen S.1 rule relaxes a protected budget (``effective_budget``:
  prompt + 64 trace slots when the prompt leaves fewer), the EFFECTIVE-budget
  ratio is reported beside the nominal one, from recorded prompt lengths.

Evidence, not trust: a cell is summarized only from rows a FINAL S.1
protection read validated, whose full dependency fingerprint still matches
(the same recomputation ``report_s.py consequence`` applies). Missing or
non-FINAL evidence gives PARTIAL; stale evidence or malformed rows give
INVALID; neither publishes a number.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

DEFINITION = (
    "nominal-budget ratio min(1, B / (prompt + generated tokens)) per row: a "
    "budget-to-sequence-length ratio, not live occupancy, K/V bytes or an "
    "eviction count"
)
RATIO_ONE = (
    "ratio 1 = the sequence never exceeded the budget (nothing pruned, by "
    "construction); ratio < 1 does not prove a prune (periodic pruning, no "
    "per-row prune record), so n_ratio_one is a LOWER bound on never-compressed"
)
RATIO_TOL = 1e-12


def summarize(values: Iterable[float]) -> dict[str, Any]:
    """Mean, median and ratio-1 count/share of per-row ratios in (0, 1]."""
    ratios = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int | float) or not 0 < v <= 1:
            raise ValueError(f"ratio out of (0, 1]: {v!r}")
        ratios.append(float(v))
    if not ratios:
        raise ValueError("no rows")
    one = sum(1 for v in ratios if v >= 1.0)
    return {
        "n_rows": len(ratios),
        "mean": statistics.fmean(ratios),
        "median": statistics.median(ratios),
        "n_ratio_one": one,
        "ratio_one_share": one / len(ratios),
    }


def prompt_lengths(tsv_bytes: bytes) -> dict[str, int]:
    """pid -> prompt length from an S.1 conditions table (one length per pid)."""
    out: dict[str, int] = {}
    for r in csv.DictReader(io.StringIO(tsv_bytes.decode()), delimiter="\t"):
        n = int(r["prompt_len"])
        if out.setdefault(r["pid"], n) != n:
            raise ValueError(f"{r['pid']}: conflicting prompt lengths")
    return out


def design_keys(tsv_bytes: bytes, press: str) -> set[tuple[str, int]]:
    """The (pid, seed) conditions the S.1 read covered for one protected press."""
    return {
        (r["pid"], int(r["seed"]))
        for r in csv.DictReader(io.StringIO(tsv_bytes.decode()), delimiter="\t")
        if r["press"] == press
    }


def s1_evidence(
    report: Mapping[str, Any] | None,
    tsv_bytes: bytes | None,
    current_fingerprint: str | None,
) -> tuple[str, list[str]]:
    """("FINAL" | "PARTIAL" | "INVALID", problems) for one budget's evidence."""
    if report is None or tsv_bytes is None:
        return "PARTIAL", ["no S.1 protection read (or its conditions table)"]
    if report.get("status") != "FINAL":
        return "PARTIAL", [f"S.1 protection read is {report.get('status')!r}"]
    problems = []
    if hashlib.sha256(tsv_bytes).hexdigest() != report.get(
        "sufficient_statistics_sha256"
    ):
        problems.append("conditions table differs from the S.1 read's hash")
    stored = (report.get("dependencies") or {}).get("sha256")
    if current_fingerprint is None or stored != current_fingerprint:
        problems.append(
            "S.1 dependency fingerprint is stale: the rows, specs or analysis "
            "code changed since the FINAL read"
        )
    return ("INVALID" if problems else "FINAL"), problems


def cell_ratios(
    rows: Sequence[Mapping[str, Any]],
    *,
    press: str,
    budget: int,
    keys: set[tuple[str, int]],
    prompt_len: Mapping[str, int],
    protected: bool,
) -> dict[str, Any]:
    """Nominal and effective ratios of one cell's rows, cross-checked.

    The rows must be exactly the S.1 design's conditions for this press (no
    duplicate, none missing, none extra), each at the nominal budget; the
    stored ratio must equal min(1, B / (prompt_len + n_tokens)) from the
    recorded prompt length (ValueError otherwise)."""
    from kvcache.presses.prompt_protect import effective_budget

    sel = [r for r in rows if r.get("press") == press]
    got = [(str(r["pid"]), int(r["decoding"]["seed"])) for r in sel]
    if len(got) != len(set(got)) or set(got) != keys:
        raise ValueError(
            f"{press}: rows do not match the S.1 design ({len(got)} rows, "
            f"{len(set(got))} unique, {len(keys)} expected)"
        )
    nominal, effective, relaxed = [], [], 0
    for r in sel:
        n = r.get("n_tokens")
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"{press}: n_tokens invalid: {n!r}")
        if int(r["budget"]) != int(budget):
            raise ValueError(f"{press}: row budget {r['budget']} is not {budget}")
        logical = prompt_len[str(r["pid"])] + n
        nom = min(1.0, budget / logical)
        stored = r.get("compression_ratio")
        if not (
            isinstance(stored, int | float)
            and not isinstance(stored, bool)
            and math.isfinite(stored)
            and abs(stored - nom) <= RATIO_TOL
        ):
            raise ValueError(
                f"{press} {r['pid']}: stored ratio {stored!r} is not "
                f"min(1, B / (prompt + n_tokens)) = {nom!r}"
            )
        target, was_relaxed = (
            effective_budget(budget, prompt_len[str(r["pid"])])
            if protected
            else (budget, False)
        )
        relaxed += int(was_relaxed)
        nominal.append(nom)
        effective.append(min(1.0, target / logical))
    out: dict[str, Any] = {**summarize(nominal), "n_budget_relaxed": relaxed}
    if relaxed:
        out["effective"] = summarize(effective)
    return out

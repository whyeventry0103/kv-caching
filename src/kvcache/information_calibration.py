"""Numerical acceptance of the information-channel calibration (Meeting 5,
M5-C; criteria frozen 2026-10-15 before any calibration data -- review of
d50aa40, item 4; protocol 6.8).

Scientific rationale. The four-block calibration is instrument development,
not evidence: it decides whether the channel can carry a confirmatory
question at all. Each check the review named is made numeric and frozen in
``configs/meeting5/info_calibration.yaml`` (``calibration_criteria``):

* C1 the full cache reads both facts out (target-query MI and correct
  probability, each query);
* C2 the answer format works (full-cache OTHER mass bounded);
* C3 the read is FINAL -- every row passed full/full identity, clone agreement
  and no-readout-eviction (the reader refuses otherwise);
* C4 the end-to-end readout (query fed into the original cache) agrees with
  the cloned readout;
* C5 eviction actually reaches the facts in the evicting arms;
* C6 dynamic range: the arms' target-query MI is not all at floor or ceiling.
  Corrected 2026-10-16, before any calibration data (review of 5616630, R3):
  the range over full + compressed arms alone passed when EVERY compressed arm
  sat at zero bits (a perfect full arm gives a 2-bit range), which can show
  catastrophic loss but cannot compare compressed policies. C6 now also needs
  at least one compressed arm at or above ``floor_mi_bits``; the
  full-to-compressed and compressed-only ranges are reported separately. No
  policy ORDER is required: equal, informative compressed arms are a valid
  null outcome.

PASS requires all six. The verdict is computed, never chosen. Validity is
separate from acceptance: malformed or missing measurements make the READ
INVALID (``kvcache.task_information.measurement_problem``) before this runs;
a complete valid channel that misses a criterion stays FINAL with calibration
FAIL. Downstream use must require ``accepted(read)``, never just exit 0.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

CRITERIA_KEYS = frozenset(
    {
        "full_min_mi_bits",
        "full_min_correct_prob",
        "full_max_other_prob",
        "max_e2e_abs_logprob_diff",
        "max_mean_early_span_retained",
        "min_mi_range_bits",
        "ceiling_mi_bits",
        "floor_mi_bits",
    }
)


def _check(value: float | None, ok: bool, rule: str) -> dict[str, Any]:
    return {"value": value, "rule": rule, "passed": bool(ok)}


def _finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def accepted(read: Mapping[str, Any]) -> bool:
    """The ONLY downstream acceptance test: a FINAL read whose frozen
    calibration evaluated to PASS."""
    cal = read.get("calibration")
    return (
        read.get("status") == "FINAL"
        and isinstance(cal, Mapping)
        and cal.get("verdict") == "PASS"
    )


def evaluate(
    read: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, float],
    *,
    full: str = "full",
    e2e_query: str | None = None,
) -> dict[str, Any]:
    """Every criterion with its value and verdict; overall PASS or FAIL.

    ``e2e_query``: the query the frozen schedule measures the end-to-end
    control on (every row of it must carry a finite value); default the last
    query present in ``rows``."""
    if set(criteria) != CRITERIA_KEYS or not all(_finite(v) for v in criteria.values()):
        raise ValueError(
            f"calibration criteria must be exactly {sorted(CRITERIA_KEYS)}, finite"
        )
    c: dict[str, dict[str, Any]] = {}
    final = read.get("status") == "FINAL"
    c["C3_read_final"] = _check(None, final, "production read is FINAL")
    if not final:
        return {"verdict": "FAIL", "criteria": c, "reason": "read is not FINAL"}
    pol = read["policies"]
    queries = [q for q in ("query_s", "query_n") if q in pol[full]]
    full_mi = min(pol[full][q]["mi_bits"]["mean"] for q in queries)
    full_cp = min(pol[full][q]["expected_correct_prob"]["mean"] for q in queries)
    c["C1_full_reads_out"] = _check(
        min(full_mi, full_cp),
        full_mi >= criteria["full_min_mi_bits"]
        and full_cp >= criteria["full_min_correct_prob"],
        f"full-cache min MI >= {criteria['full_min_mi_bits']} bits and min "
        f"correct prob >= {criteria['full_min_correct_prob']} (MI {full_mi:.4f}, "
        f"p {full_cp:.4f})",
    )
    other = max(pol[full][q]["other_prob"]["mean"] for q in queries)
    c["C2_format"] = _check(
        other,
        other <= criteria["full_max_other_prob"],
        f"full-cache OTHER mass <= {criteria['full_max_other_prob']}",
    )
    # C4: EVERY scheduled measurement must be present and finite; a subset, a
    # None or a NaN fails (the reader already makes such a read INVALID --
    # this is the evaluator refusing to be the weaker link)
    e2e_q = e2e_query or (rows[-1]["query"] if rows else None)
    scheduled = [r.get("e2e_max_abs_diff") for r in rows if r.get("query") == e2e_q]
    complete = bool(scheduled) and all(_finite(v) and v >= 0 for v in scheduled)
    worst = max(float(v) for v in scheduled) if complete else None
    c["C4_end_to_end"] = _check(
        worst,
        complete and worst <= criteria["max_e2e_abs_logprob_diff"],
        f"max |end-to-end - cloned| log-prob <= "
        f"{criteria['max_e2e_abs_logprob_diff']} over all {len(scheduled)} "
        f"scheduled readouts (query {e2e_q!r})",
    )
    early: list[float] = []
    early_ok = True
    for r in rows:
        if r["policy"] == full:
            continue
        spans = r["fact_spans"]
        first = min(spans, key=lambda k: spans[k][0])
        frac = (r.get("retained_fraction") or {}).get(first)
        if not (_finite(frac) and 0.0 <= frac <= 1.0):
            early_ok = False  # missing evidence cannot drop out of the mean
            continue
        early.append(float(frac))
    mean_early = statistics.fmean(early) if early and early_ok else None
    c["C5_eviction_reaches_facts"] = _check(
        mean_early,
        mean_early is not None
        and mean_early <= criteria["max_mean_early_span_retained"],
        f"evicting arms' mean retained fraction of the early fact <= "
        f"{criteria['max_mean_early_span_retained']} (every evicting row measured)",
    )
    target = {p: pol[p]["query_s"]["mi_bits"]["mean"] for p in pol}
    evicting = [v for p, v in target.items() if p != full]
    spread = max(target.values()) - min(target.values())
    not_ceiling = not all(v >= criteria["ceiling_mi_bits"] for v in evicting)
    not_floor = bool(evicting) and max(evicting) >= criteria["floor_mi_bits"]
    c6 = _check(
        spread,
        spread >= criteria["min_mi_range_bits"] and not_ceiling and not_floor,
        f"target-query MI range across arms >= {criteria['min_mi_range_bits']} "
        f"bits, not every compressed arm >= {criteria['ceiling_mi_bits']} bits, "
        f"and at least one compressed arm >= {criteria['floor_mi_bits']} bits "
        "(no policy order is required)",
    )
    c6["ranges_bits"] = {
        "full_to_compressed": (
            target[full] - min(evicting) if evicting and full in target else None
        ),
        "compressed_only": max(evicting) - min(evicting) if evicting else None,
        "all_arms": spread,
        "max_compressed": max(evicting) if evicting else None,
    }
    c["C6_dynamic_range"] = c6
    verdict = "PASS" if all(v["passed"] for v in c.values()) else "FAIL"
    return {
        "verdict": verdict,
        "criteria": c,
        "failed": sorted(k for k, v in c.items() if not v["passed"]),
        "note": "calibration is instrument development, never confirmatory",
    }

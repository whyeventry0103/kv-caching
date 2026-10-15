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

PASS requires all six. The verdict is computed, never chosen.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def _check(value: float | None, ok: bool, rule: str) -> dict[str, Any]:
    return {"value": value, "rule": rule, "passed": bool(ok)}


def evaluate(
    read: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, float],
    *,
    full: str = "full",
) -> dict[str, Any]:
    """Every criterion with its value and verdict; overall PASS or FAIL."""
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
    diffs = [
        float(r["e2e_max_abs_diff"])
        for r in rows
        if r.get("e2e_max_abs_diff") is not None
    ]
    worst = max(diffs) if diffs else math.inf
    c["C4_end_to_end"] = _check(
        worst,
        bool(diffs) and worst <= criteria["max_e2e_abs_logprob_diff"],
        f"max |end-to-end - cloned| log-prob <= "
        f"{criteria['max_e2e_abs_logprob_diff']} over {len(diffs)} readouts",
    )
    early = []
    for r in rows:
        if r["policy"] == full or not r.get("retained_fraction"):
            continue
        spans = r["fact_spans"]
        first = min(spans, key=lambda k: spans[k][0])
        frac = r["retained_fraction"].get(first)
        if frac is not None:
            early.append(float(frac))
    mean_early = statistics.fmean(early) if early else None
    c["C5_eviction_reaches_facts"] = _check(
        mean_early,
        mean_early is not None
        and mean_early <= criteria["max_mean_early_span_retained"],
        f"evicting arms' mean retained fraction of the early fact <= "
        f"{criteria['max_mean_early_span_retained']}",
    )
    target = {p: pol[p]["query_s"]["mi_bits"]["mean"] for p in pol}
    spread = max(target.values()) - min(target.values())
    evicting = [v for p, v in target.items() if p != full]
    not_ceiling = not all(v >= criteria["ceiling_mi_bits"] for v in evicting)
    c["C6_dynamic_range"] = _check(
        spread,
        spread >= criteria["min_mi_range_bits"] and not_ceiling,
        f"target-query MI range across arms >= {criteria['min_mi_range_bits']} "
        f"bits and not every evicting arm >= {criteria['ceiling_mi_bits']}",
    )
    verdict = "PASS" if all(v["passed"] for v in c.values()) else "FAIL"
    return {
        "verdict": verdict,
        "criteria": c,
        "failed": sorted(k for k, v in c.items() if not v["passed"]),
        "note": "calibration is instrument development, never confirmatory",
    }

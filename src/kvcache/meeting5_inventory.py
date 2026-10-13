"""Inventory of reusable rows, strict-fit cohort and row-count/cost ledger for
the Meeting 5 manifest (prepared 2026-10-14; read-only over existing runs).

Scientific rationale. The Meeting 5 review rejected the old WP5 budget as a
block: before any new compute is requested, every cell of the matched
prompt-protection comparison must be classified as REUSABLE (complete, valid
existing rows on the frozen problems and seeds), or NEW (rows that must be
generated), and the price of the NEW cells must come from measured timings of
comparable existing rows, not from a stale rate.

* Coverage is judged against the EXPECTED keys from the frozen problem list and
  the configured seeds, never against whatever rows exist (duplicates and
  foreign keys are reported, not absorbed).
* The strict-fit cohort is the problems whose prompt leaves at least 64 trace
  slots at the nominal budget (P + 64 <= B), so that no policy's arm receives
  extra capacity from the S.1 relaxation; it is computed from the full-cache
  bank's prompt token ids, before any new outcome.
* Cost: a cell's total time is n_rows x the MEAN per-row ``wallclock_s``
  (row times are right-skewed by long and truncated generations, so a median
  would underprice the sum). The policy changes trace length, which dominates
  the uncertainty, so the band is the slowest and fastest MEAN among the
  comparable existing cells (same model, benchmark, budget). Model-loading and
  retry allowances are added explicitly.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

MIN_TRACE_SLOTS = 64
LOAD_HOURS_PER_LAUNCH = 0.05  # model load + warm-up, measured order (3 min)
RETRY_FRACTION = 0.10  # allowance for failed/retried shards


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Result rows of one run directory (token-id and log files excluded)."""
    rows = []
    for path in sorted(Path(run_dir).rglob("*.jsonl")):
        if "token_ids" in path.parts or "logs" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["model"]),
        str(row["bench"]),
        str(row["press"]).lower(),
        str(row["budget"]),
    )


def inventory_run(
    run: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_pids: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """One entry per (model, bench, press, budget) cell found in a run."""
    cells: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    want = {(p, int(s)) for p in expected_pids for s in expected_seeds}
    out = []
    for key, rs in sorted(cells.items()):
        keys = [(str(r["pid"]), int(r["decoding"]["seed"])) for r in rs]
        have = set(keys)
        walls = sorted(float(r["wallclock_s"]) for r in rs if "wallclock_s" in r)
        out.append(
            {
                "run": run,
                "model": key[0],
                "bench": key[1],
                "press": key[2],
                "budget": key[3],
                "n_rows": len(rs),
                "n_expected": len(want),
                "n_covered": len(have & want),
                "n_duplicates": len(keys) - len(have),
                "n_foreign": len(have - want),
                "complete": want <= have and len(keys) == len(have),
                "n_truncated": sum(bool(r.get("truncated")) for r in rs),
                "wallclock_s": rate_band(walls),
            }
        )
    return out


def rate_band(walls: Sequence[float]) -> dict[str, float] | None:
    """Mean, median and quartiles of per-row seconds (None if < 4 timings)."""
    if len(walls) < 4:
        return None
    q = statistics.quantiles(walls, n=4)
    return {
        "mean": statistics.fmean(walls),
        "median": statistics.median(walls),
        "q1": q[0],
        "q3": q[2],
        "n": float(len(walls)),
    }


def strict_fit(
    prompt_lens: Mapping[str, int], budget: int, min_trace: int = MIN_TRACE_SLOTS
) -> dict[str, Any]:
    """Problems with P + min_trace <= budget, and the relaxed ones with B_eff."""
    fit = sorted(p for p, n in prompt_lens.items() if n + min_trace <= budget)
    relaxed = {
        p: n + min_trace
        for p, n in sorted(prompt_lens.items())
        if n + min_trace > budget
    }
    return {
        "budget": budget,
        "min_trace_slots": min_trace,
        "n_problems": len(prompt_lens),
        "n_strict_fit": len(fit),
        "strict_fit_pids": fit,
        "relaxed_effective_budgets": relaxed,
    }


def comparable_rates(
    cells: Iterable[Mapping[str, Any]], *, model: str, bench: str, budget: str
) -> dict[str, Any]:
    """Fastest / pooled / slowest mean row time over complete comparable cells."""
    means = {
        (c["run"], c["press"]): c["wallclock_s"]["mean"]
        for c in cells
        if c["model"] == model
        and c["bench"] == bench
        and str(c["budget"]) == str(budget)
        and c["complete"]
        and c["wallclock_s"]
    }
    if not means:
        raise ValueError(f"no complete comparable cell for {model}/{bench}/{budget}")
    return {
        "lower_s": min(means.values()),
        "central_s": statistics.fmean(means.values()),
        "upper_s": max(means.values()),
        "cells": sorted(f"{r}:{p}" for r, p in means),
    }


def cost_hours(
    n_rows: int, rates: Mapping[str, float], *, n_launches: int
) -> dict[str, float]:
    """Lower / central / upper allocated GPU-hours for ``n_rows`` new rows."""

    def h(sec: float) -> float:
        base = n_rows * sec / 3600 + n_launches * LOAD_HOURS_PER_LAUNCH
        return round(base * (1 + RETRY_FRACTION), 2)

    return {
        "lower": h(rates["lower_s"]),
        "central": h(rates["central_s"]),
        "upper": h(rates["upper_s"]),
    }


def inventory_readout_run(
    run: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_pids: Sequence[str],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Readout rows (schema: reference_key + condition): coverage per condition.

    The key is the reference trace's (pid, seed); ``n_defined`` counts rows
    whose readout boundary was defined (an undefined row stays in the count).
    """
    cells: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        ref = r["reference_key"]
        cond = r["condition"]
        key = (str(ref[2]), str(cond["press"]).lower(), str(cond["budget"]))
        cells.setdefault(key, []).append(r)
    want = {(p, int(s)) for p in expected_pids for s in expected_seeds}
    out = []
    for (model, press, budget), rs in sorted(cells.items()):
        keys = [(str(r["reference_key"][1]), int(r["reference_key"][5])) for r in rs]
        have = set(keys)
        out.append(
            {
                "run": run,
                "kind": "readout",
                "model": model,
                "press": press,
                "budget": budget,
                "n_rows": len(rs),
                "n_expected": len(want),
                "n_covered": len(have & want),
                "n_duplicates": len(keys) - len(have),
                "n_foreign": len(have - want),
                "complete": want <= have and len(keys) == len(have),
                "n_defined": sum(bool(r.get("readout_defined")) for r in rs),
            }
        )
    return out


def hashed_split(
    ids: Sequence[str],
    *,
    salt: str,
    n_calibration: int,
    n_confirmation: int,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """Disjoint calibration / confirmation / reserve lists by SHA-256 order.

    The order depends only on (salt, id) -- never on any outcome -- so the
    split is reproducible from the id list alone. ``exclude`` removes ids that
    have already been used (e.g. the discovery cohort) before ordering.
    """
    import hashlib

    pool = sorted(set(ids) - set(exclude))
    if len(pool) != len(set(ids)) - len(set(exclude) & set(ids)):
        raise ValueError("duplicate ids")
    if n_calibration + n_confirmation > len(pool):
        raise ValueError(f"only {len(pool)} ids for {n_calibration}+{n_confirmation}")
    order = sorted(
        pool, key=lambda x: hashlib.sha256(f"{salt}|{x}".encode()).hexdigest()
    )
    cal = order[:n_calibration]
    conf = order[n_calibration : n_calibration + n_confirmation]
    return {
        "salt": salt,
        "n_pool": len(pool),
        "n_excluded": len(set(exclude) & set(ids)),
        "calibration": sorted(cal),
        "confirmation": sorted(conf),
        "reserve": sorted(order[n_calibration + n_confirmation :]),
    }

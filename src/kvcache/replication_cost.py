"""Cost model for the M5-B multi-dataset replication (prepared 2026-11-13).

Scientific rationale. A MATH-500 row time does not transfer to GSM8K or BBH:
row time is dominated by trace length, and the new tasks' traces are far
shorter than MATH-500's. The model therefore separates two measured inputs:

* per-press **seconds per generated token** and **length inflation** (a
  compressed trace's tokens over the matched full-cache trace's tokens), both
  from the complete MATH-500 Qwen-7B B=256 cells (``press_rates``); and
* the new dataset's **full-cache trace lengths and seconds per token**, from
  its own calibration bank (the timing read).

A compressed row is priced as ``min(cap, inflation * T_fc) * s_per_token``
per full-cache trace (central), with the all-at-cap row as the upper bound. A
forward readout replays the reference trace and generates an answer of at most
``max_answer_tokens``; its rate per (replayed + answer) token is calibrated on
the settled P1 campaign (``readout_rate``). Allocated GPU-hours add model
loading per launch and the Meeting 5 retry allowance
(``meeting5_inventory.cost_hours`` conventions), so caps cover what the
campaign ledger actually charges. The model prices; it never selects a design.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kvcache.meeting5_inventory import LOAD_HOURS_PER_LAUNCH, RETRY_FRACTION


def press_rates(
    rows: Iterable[Mapping[str, Any]], fc_tokens: Mapping[tuple[str, int], int]
) -> dict[str, dict[str, float]]:
    """Per press: pooled s/token, mean tokens, matched length inflation over
    the full-cache bank (sum of compressed tokens / sum of matched FC tokens),
    and truncation rate. Rows without a matched FC trace are not inflated."""
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["press"]), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for press, rs in sorted(by.items()):
        tok = sum(int(r["n_tokens"]) for r in rs)
        wall = sum(float(r["wallclock_s"]) for r in rs)
        matched = [
            (int(r["n_tokens"]), int(fc_tokens[(str(r["pid"]), int(r["seed"]))]))
            for r in rs
            if (str(r["pid"]), int(r["seed"])) in fc_tokens
        ]
        if not matched or not tok:
            raise ValueError(f"{press}: no matched full-cache trace or no tokens")
        out[press] = {
            "n": float(len(rs)),
            "s_per_token": wall / tok,
            "mean_tokens": tok / len(rs),
            "inflation": sum(a for a, _ in matched) / sum(b for _, b in matched),
            "n_matched": float(len(matched)),
            "truncation_rate": sum(bool(r["truncated"]) for r in rs) / len(rs),
        }
    return out


def readout_rate(
    settled_gpu_hours: float, n_records: int, mean_replayed: float, mean_answer: float
) -> float:
    """Seconds per (replayed + answer) token from a settled readout campaign."""
    if n_records <= 0 or mean_replayed + mean_answer <= 0:
        raise ValueError("empty readout calibration")
    return settled_gpu_hours * 3600 / (n_records * (mean_replayed + mean_answer))


def _alloc(seconds: float, n_launches: int) -> float:
    return round(
        (seconds / 3600 + n_launches * LOAD_HOURS_PER_LAUNCH) * (1 + RETRY_FRACTION),
        2,
    )


def grid_cost(
    fc_lengths: Sequence[int],
    *,
    n_rows: int,
    s_per_token: float,
    inflation: float,
    cap: int,
    n_launches: int,
) -> dict[str, float]:
    """Central and upper allocated GPU-hours for ``n_rows`` compressed rows
    whose full-cache traces have ``fc_lengths`` (the calibration bank)."""
    if not fc_lengths:
        raise ValueError("no full-cache lengths")
    per_trace = statistics.fmean(min(cap, inflation * t) for t in fc_lengths)
    return {
        "central": _alloc(n_rows * per_trace * s_per_token, n_launches),
        "upper": _alloc(n_rows * cap * s_per_token, n_launches),
        "mean_tokens_central": per_trace,
    }


def readout_cost(
    fc_lengths: Sequence[int],
    *,
    n_records: int,
    s_per_token: float,
    mean_answer: float,
    max_answer_tokens: int,
    n_launches: int,
) -> dict[str, float]:
    """Central (mean answer) and upper (every answer at its cap) GPU-hours for
    forward readouts replaying full-cache traces of ``fc_lengths``."""
    if not fc_lengths:
        raise ValueError("no full-cache lengths")
    replay = statistics.fmean(fc_lengths)
    return {
        "central": _alloc(n_records * (replay + mean_answer) * s_per_token, n_launches),
        "upper": _alloc(
            n_records * (replay + max_answer_tokens) * s_per_token, n_launches
        ),
    }


WINDOW_GPU_HOURS = 2 * (8 + 58 / 60)  # two GPUs x the 01:45-10:43 Dubai window
KB_PER_ROW = {"grid": 90.0, "bank": 160.0, "readout": 8.0}  # measured, rounded up


def _launches(hours: float) -> int:
    """Two shards per night-window the cell occupies (at least one night)."""
    return 2 * max(1, -(-int(hours * 10) // int(WINDOW_GPU_HOURS * 10)))


def plan_stages(
    design: Mapping[str, Any],
    rates: Mapping[str, Any],
    timing: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Price every cell of every stage in ``design`` (see
    ``configs/meeting5/replication_design_m5b.json``) for each dataset whose
    timing read is given; cumulative caps are the stage's upper hours."""
    out: dict[str, Any] = {"stages": [], "rates_source": rates.get("source")}
    fwd = float(rates["readout"]["forward"]["s_per_token"])
    rev = max(fwd, float(rates["readout"]["reverse"]["s_per_token"]))
    answer = float(rates["readout"]["forward"]["mean_answer_tokens"])
    for stage in design["stages"]:
        bench = stage["benchmark"]
        if bench not in timing:
            out["stages"].append({"id": stage["id"], "status": "UNPRICED (no timing)"})
            continue
        t = timing[bench]
        if t.get("cap_bound"):
            # no calibration bank yet: every trace at the cap (an upper bound
            # only; central == upper), with the given full-cache rate
            t = {
                "conditions": [{"n_tokens": int(stage["max_new_tokens"])}],
                "seconds_per_token": float(t["seconds_per_token"]),
            }
        lengths = [int(c["n_tokens"]) for c in t["conditions"]]
        n_unit = int(stage["n_problems"]) * int(stage["n_seeds"])
        cells = []
        for cell in stage["cells"]:
            kind, n = cell["kind"], n_unit * len(cell.get("presses", [None]))
            if kind == "bank":
                c = grid_cost(
                    lengths,
                    n_rows=n,
                    s_per_token=float(t["seconds_per_token"]),
                    inflation=1.0,
                    cap=int(stage["max_new_tokens"]),
                    n_launches=2,
                )
            elif kind == "grid":
                c = {"central": 0.0, "upper": 0.0}
                for p in cell["presses"]:
                    r = rates["presses"][p]
                    pc = grid_cost(
                        lengths,
                        n_rows=n_unit,
                        s_per_token=float(r["s_per_token"]),
                        inflation=float(r["inflation"]),
                        cap=int(stage["max_new_tokens"]),
                        n_launches=0,
                    )
                    c["central"] += pc["central"]
                    c["upper"] += pc["upper"]
            elif kind in ("readout", "reverse"):
                rep = lengths
                if kind == "reverse":
                    inf = max(
                        float(rates["presses"][p]["inflation"]) for p in cell["presses"]
                    )
                    rep = [
                        min(int(stage["max_new_tokens"]), int(inf * x)) for x in lengths
                    ]
                c = readout_cost(
                    rep,
                    n_records=n,
                    s_per_token=fwd if kind == "readout" else rev,
                    mean_answer=answer,
                    max_answer_tokens=int(stage["max_answer_tokens"]),
                    n_launches=0,
                )
            else:
                raise ValueError(f"unknown cell kind {kind!r}")
            if kind != "bank":
                load = _launches(c["central"]) * LOAD_HOURS_PER_LAUNCH
                c = {
                    k: round(v + load * (1 + RETRY_FRACTION), 2)
                    for k, v in c.items()
                    if k in ("central", "upper")
                }
            kb = KB_PER_ROW["readout" if kind in ("readout", "reverse") else kind]
            cells.append(
                {
                    **cell,
                    "rows": n,
                    "gpu_hours": c,
                    "storage_gb": round(n * kb / 1e6, 3),
                }
            )
        central = round(sum(c["gpu_hours"]["central"] for c in cells), 2)
        upper = round(sum(c["gpu_hours"]["upper"] for c in cells), 2)
        out["stages"].append(
            {
                "id": stage["id"],
                "benchmark": bench,
                "answers": stage["answers"],
                "priced_from": (
                    "cap bound (no calibration bank)"
                    if timing[bench].get("cap_bound")
                    else "own calibration bank"
                ),
                "cells": cells,
                "rows": sum(c["rows"] for c in cells),
                "gpu_hours_central": central,
                "gpu_hours_upper": upper,
                "nights_central": round(central / WINDOW_GPU_HOURS, 1),
                "nights_upper": round(upper / WINDOW_GPU_HOURS, 1),
                "storage_gb": round(sum(c["storage_gb"] for c in cells), 3),
            }
        )
    return out

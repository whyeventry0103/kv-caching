"""Coverage checks and registered GSM8K scoring for frozen-ID compressed runs
and forward readouts (M5-B replication paths; prepared 2026-11-10). The
replication's estimands, inference and decomposition are NOT here: they are
fixed with the design after the timing read.

Scientific rationale. Two failure modes would silently bias a replication
read: scoring a different set of conditions than the design (a missing,
repeated or foreign key; an orphan row; an unpublished envelope), and scoring
GSM8K answers with the MATH grader stored in generation rows and readout
records. Every compressed-run row therefore passes the same pair validation
as the timing bank (``gsm8k_timing.validate_pairs``), and every score -- free
run and readout alike -- comes from the registered read-time scorer
(``task_scoring.rescore`` via ``gsm8k_timing.score_rows``) on the stored text.

Readout convention (proposed with these paths; confirm with the design before
any read): a readout record whose reasoning/answer boundary is undefined is
excluded from readout accuracy and counted, never dropped or scored, exactly
as in the MATH-500 readouts; a defined readout whose generated answer span
reached ``max_answer_tokens`` is scored as truncated (incorrect), the GSM8K
rule for an unfinished answer. The record's own ``readout_correct`` (MATH
grader) is never the score.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kvcache.gsm8k_timing import score_rows, validate_pairs
from kvcache.manifest_grid import design_keys

ReadoutKey = tuple[str, str, str, int]  # (press, budget, pid, seed)


def _cell_summary(scores: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_pid: dict[str, list[bool]] = {}
    for s in scores:
        by_pid.setdefault(str(s["pid"]), []).append(bool(s["correct"]))
    return {
        "n": len(scores),
        "accuracy": sum(s["correct"] for s in scores) / len(scores),
        "accuracy_equal_problem": statistics.fmean(
            statistics.fmean(v) for v in by_pid.values()
        ),
        "truncation_rate": sum(bool(s["truncated"]) for s in scores) / len(scores),
    }


def score_grid(
    runs_root: Path,
    design: Mapping[str, Any],
    golds: Mapping[str, str],
) -> dict[str, Any]:
    """Every expected compressed-run pair validated and scored, per cell."""
    split = {"split": design["split"], "revision": design["revision"]}
    pairs = validate_pairs(
        runs_root,
        bench=design["benchmark"],
        keys=design_keys(design),
        manifest_split=split,
    )
    scored = score_rows(
        [r for r, _ in pairs],
        golds,
        revision=design["revision"],
        bench=design["benchmark"],
    )
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for s in scored["scores"]:
        cells.setdefault(f"{s['press']}-{s['budget']}", []).append(s)
    return {
        "status": "FINAL",
        "validated_pairs": len(pairs),
        "grader": scored["grader"],
        "grader_sha256": scored["grader_sha256"],
        "cells": {k: _cell_summary(v) for k, v in sorted(cells.items())},
        "scores": scored["scores"],
    }


def _readout_records(runs_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, torn = [], []
    base = Path(runs_root) / "readout"
    for path in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                torn.append(f"{path.name}:{n}")
    return records, torn


def _readout_key(rec: Mapping[str, Any]) -> ReadoutKey:
    ref = rec["reference_key"]
    cond = rec["condition"]
    return (str(cond["press"]), str(cond["budget"]), str(ref[1]), int(ref[5]))


def readout_rows(
    records: Iterable[Mapping[str, Any]], *, max_answer_tokens: int
) -> list[dict[str, Any]]:
    """Defined readout records as scorer rows (the generated answer span is the
    trace; reaching the answer cap is truncation)."""
    rows = []
    for rec in records:
        press, budget, pid, seed = _readout_key(rec)
        rows.append(
            {
                "pid": pid,
                "press": press,
                "budget": budget,
                "decoding": {"seed": seed},
                "trace": rec["generated_text"],
                "truncated": len(rec["generated_token_ids"]) >= max_answer_tokens,
                "correct": rec.get("readout_correct"),
            }
        )
    return rows


def score_readouts(
    runs_root: Path,
    design: Mapping[str, Any],
    golds: Mapping[str, str],
) -> dict[str, Any]:
    """Exactly one readout record per expected key, then registered scores of
    the defined ones, per cell; ValueError on any coverage problem."""
    records, torn = _readout_records(runs_root)
    want = set(design_keys(design))
    errors: list[str] = []
    if torn:
        errors.append(f"torn readout lines: {torn[:3]}")
    by_key: dict[ReadoutKey, list[dict[str, Any]]] = {}
    for rec in records:
        ref = rec.get("reference_key") or []
        if (
            len(ref) != 6
            or ref[0] != design["benchmark"]
            or ref[2] != design["model"]
            or (str(ref[3]), str(ref[4])) != ("fullcache", "full")
        ):
            errors.append(f"readout not of this design's full-cache bank: {ref}")
            continue
        by_key.setdefault(_readout_key(rec), []).append(rec)
    missing = sorted(want - set(by_key))
    extra = sorted(set(by_key) - want)
    dup = sorted(k for k, v in by_key.items() if len(v) > 1)
    if missing:
        errors.append(f"{len(missing)} expected readouts missing: {missing[0]}")
    if extra:
        errors.append(f"{len(extra)} readouts outside the design: {extra[0]}")
    if dup:
        errors.append(f"{len(dup)} readouts repeated: {dup[0]}")
    if errors:
        raise ValueError("; ".join(errors[:4]))
    ordered = [by_key[k][0] for k in design_keys(design)]
    defined = [r for r in ordered if r.get("readout_defined") is True]
    undefined = [r for r in ordered if r.get("readout_defined") is not True]
    scored = score_rows(
        readout_rows(defined, max_answer_tokens=int(design["max_answer_tokens"])),
        golds,
        revision=design["revision"],
        bench=design["benchmark"],
    )
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for s in scored["scores"]:
        cells.setdefault(f"{s['press']}-{s['budget']}", []).append(s)
    undefined_by_cell: dict[str, int] = {}
    for r in undefined:
        press, budget, _, _ = _readout_key(r)
        key = f"{press}-{budget}"
        undefined_by_cell[key] = undefined_by_cell.get(key, 0) + 1
    return {
        "status": "FINAL",
        "expected": len(want),
        "defined": len(defined),
        "undefined": len(undefined),
        "undefined_by_cell": dict(sorted(undefined_by_cell.items())),
        "grader": scored["grader"],
        "grader_sha256": scored["grader_sha256"],
        "cells": {k: _cell_summary(v) for k, v in sorted(cells.items())},
        "scores": scored["scores"],
    }

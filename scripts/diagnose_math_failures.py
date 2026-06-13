#!/usr/bin/env python3
"""Non-generative MATH failure / canonicalization audit (Milestone A).

Reads an existing full-cache run directory of schema JSONL rows and, for every
MATH-500 cell, re-scores the answer with the vetted grader
(:func:`kvcache.grading.is_equiv`) against the *raw* gold, classifies each failure,
and quantifies how many rows the old canonical-string scoring got wrong.

This script never loads a model and never touches a GPU. Gold answers are not
stored in the rows, so they come from either ``--gold-map`` (a JSON mapping
``pid -> raw gold``) or, on the remote where MATH-500 is cached, from the pinned
dataset via ``--gold-from-dataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.generation import has_answer_marker  # noqa: E402
from kvcache.grading import GRADER_VERSION, is_equiv  # noqa: E402

TRACE_TAIL_CHARS = 800


def load_rows(rundir: Path, *, bench: str = "math500") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(rundir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bench") == bench:
                rows.append(row)
    return rows


def classify_row(row: dict[str, Any], gold_raw: str | None) -> dict[str, Any]:
    """Re-score one row and bucket it into a failure-mode classification."""
    trace = row.get("trace", "")
    answer_raw = row.get("answer_raw", "")
    max_new = row["decoding"]["max_new"]
    hit_token_cap = row["n_tokens"] >= max_new
    marker = has_answer_marker(trace)
    truncated = bool(row.get("truncated"))
    stored_correct = bool(row.get("correct"))
    ref_correct = is_equiv(gold_raw, answer_raw) if gold_raw is not None else None

    if ref_correct:
        classification = "correct"
    elif truncated or (hit_token_cap and not marker):
        classification = "token_cap_truncation"
    elif not marker:
        classification = "extraction_failure"
    elif gold_raw is None:
        classification = "gold_unavailable"
    else:
        # Marker present, finished under cap, but graded wrong. We cannot
        # auto-distinguish a genuine wrong answer from an ambiguous-gold
        # equivalence the grader misses; flag for human read.
        classification = "genuine_wrong_or_ambiguous"

    grader_disagreement = (
        None if ref_correct is None else (stored_correct != ref_correct)
    )
    return {
        "pid": row.get("pid"),
        "seed": row["decoding"]["seed"],
        "level": row.get("runmeta", {}).get("level")
        or row.get("level")
        or row.get("meta", {}).get("level"),
        "n_tokens": row["n_tokens"],
        "max_new": max_new,
        "hit_token_cap": hit_token_cap,
        "has_answer_marker": marker,
        "truncated": truncated,
        "gold_raw": gold_raw,
        "answer_raw": answer_raw,
        "answer_canon": row.get("answer_canon"),
        "stored_correct": stored_correct,
        "ref_correct": ref_correct,
        "grader_disagreement": grader_disagreement,
        "classification": classification,
        "trace_tail": trace[-TRACE_TAIL_CHARS:],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(record["classification"] for record in records)
    false_neg = sum(
        1
        for record in records
        if record["grader_disagreement"] and not record["stored_correct"]
    )
    false_pos = sum(
        1
        for record in records
        if record["grader_disagreement"] and record["stored_correct"]
    )
    return {
        "rows": len(records),
        "classifications": dict(classifications),
        "grader_false_negatives": false_neg,
        "grader_false_positives": false_pos,
        "grader_disagreements": false_neg + false_pos,
        "grader_version": GRADER_VERSION,
        "stored_accuracy": (
            sum(record["stored_correct"] for record in records) / len(records)
            if records
            else 0.0
        ),
        "regraded_accuracy": (
            sum(bool(record["ref_correct"]) for record in records) / len(records)
            if records
            else 0.0
        ),
    }


def render_markdown(records: list[dict[str, Any]]) -> str:
    header = (
        "| pid | seed | n_tok | cap | marker | stored | regraded | disagree | "
        "class |\n|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    ordered = sorted(records, key=lambda record: (str(record["pid"]), record["seed"]))
    for record in ordered:
        lines.append(
            "| {pid} | {seed} | {n_tokens} | {hit_token_cap} | {has_answer_marker} | "
            "{stored_correct} | {ref_correct} | {grader_disagreement} | "
            "{classification} |".format(**record)
        )
    return "\n".join(lines)


def _gold_map_from_dataset() -> dict[str, str]:
    import yaml

    from kvcache.datasets import load_benchmark

    benchmarks = yaml.safe_load(
        (PROJECT_ROOT / "configs/benchmarks.yaml").read_text(encoding="utf-8")
    )["benchmarks"]
    problems = load_benchmark("math500", benchmarks["math500"], subset=None)
    return {problem.pid: problem.gold_answer for problem in problems}


def main() -> int:
    parser = argparse.ArgumentParser(description="MATH failure / canon audit")
    parser.add_argument("--rundir", required=True, type=Path)
    parser.add_argument("--gold-map", type=Path, help="JSON: pid -> raw gold")
    parser.add_argument("--gold-from-dataset", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.gold_map:
        gold_map = json.loads(args.gold_map.read_text(encoding="utf-8"))
    elif args.gold_from_dataset:
        gold_map = _gold_map_from_dataset()
    else:
        gold_map = {}

    rows = load_rows(args.rundir)
    records = [classify_row(row, gold_map.get(row.get("pid"))) for row in rows]
    summary = summarize(records)

    out = args.out or (PROJECT_ROOT / "results/diag" / args.rundir.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "table.md").write_text(render_markdown(records), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {out}/records.json, summary.json, table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from kvcache.grading import (  # noqa: E402
    CROSSCHECK_VERSION,
    GRADER_VERSION,
    crosscheck_equiv,
    is_equiv,
)

TRACE_TAIL_CHARS = 800


def load_rows(rundir: Path, *, bench: str = "math500") -> list[dict[str, Any]]:
    """Result rows under ``rundir``, excluding token-ID sidecars.

    Sidecars live in a ``token_ids/`` subtree, are also ``*.jsonl``, and also
    carry ``bench`` -- so the bench filter alone lets them through, and they
    have no ``decoding`` block. This script predates sidecars: it was written
    against the full-cache bank and worked, then silently acquired a crash the
    first time it was pointed at a compression cell.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(rundir.rglob("*.jsonl")):
        if "token_ids" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bench") == bench and "decoding" in row:
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
    # Re-running is_equiv over stored answers agrees with itself by
    # construction: same normalization, same inputs. That measures replay
    # determinism, NOT grader validity, and cannot detect the threat this audit
    # exists for -- that is_equiv under-credits equivalent forms more often on
    # the malformed generations compression produces. The cross-check is a
    # different normalization (sympify canonical form) so a disagreement is
    # informative. Neither grader wins automatically; disagreements are
    # candidates for human review.
    cross_correct = crosscheck_equiv(gold_raw, answer_raw)
    cross_disagreement = (
        None
        if (cross_correct is None or ref_correct is None)
        else (bool(cross_correct) != bool(ref_correct))
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
        "cross_correct": cross_correct,
        "cross_disagreement": cross_disagreement,
        "classification": classification,
        "trace_tail": trace[-TRACE_TAIL_CHARS:],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(record["classification"] for record in records)
    cross_comparable = sum(1 for r in records if r["cross_disagreement"] is not None)
    cross_disagree = sum(1 for r in records if r["cross_disagreement"])
    # Rows the cross-check calls correct and is_equiv calls wrong: the
    # under-crediting direction, which is the one that would bias every
    # accuracy change toward larger apparent compression damage.
    cross_only = sum(1 for r in records if r["cross_correct"] and not r["ref_correct"])
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
        "crosscheck_version": CROSSCHECK_VERSION,
        "crosscheck_comparable": cross_comparable,
        "crosscheck_disagreements": cross_disagree,
        "crosscheck_only_correct": cross_only,
        "crosscheck_note": (
            "grader_disagreements re-runs is_equiv over stored answers and "
            "therefore agrees by construction: it is a REPLAY DETERMINISM "
            "check, not grader validity. crosscheck_* uses a different "
            "normalization (sympify canonical form) and is the only column "
            "here that speaks to whether is_equiv under-credits."
        ),
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

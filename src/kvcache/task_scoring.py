"""Dataset-aware read-time scoring for the Meeting 5 replication tasks (review
of 6b409001, R4; prepared 2026-10-14, before any GSM8K or BBH generation).

Scientific rationale. Generation rows carry an in-row ``correct`` computed by
the MATH grader, which is wrong for GSM8K-style numbers (``1,250``, ``18.00``,
``72 clips``) and was never designed for BBH options. Rather than edit the
generation path (generation identity of every pending cell) or re-score
historical MATH rows (frozen), the primary score of a GSM8K or BBH row is
computed HERE, at read time, from the stored raw trace:

* dispatch is by benchmark AND its frozen revision; the grader version and the
  sha256 of the grader source are recorded, so a grader change is visible in
  every score;
* every arm -- full cache and every compressed arm -- goes through the same
  function and the same truncation rule (truncated = incorrect);
* each score records the sha256 of the raw trace it was computed from;
* MATH-500 is refused: its frozen primary is the in-row grader.

A row without a boolean ``truncated``, a string trace, or a gold answer makes
the whole re-score INVALID; nothing is silently dropped.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRADER_SOURCE = "src/kvcache/task_graders.py"
FINAL, INVALID = "FINAL", "INVALID"


def registry(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Benchmark -> frozen revision, grader version and grading function."""
    import yaml

    from kvcache import bbh
    from kvcache.task_graders import GRADER_VERSIONS, bbh_grade, gsm8k_grade

    base = Path(root) if root is not None else PROJECT_ROOT
    benches = yaml.safe_load((base / "configs/benchmarks.yaml").read_text())
    return {
        "gsm8k": {
            "revision": str(benches["benchmarks"]["gsm8k"]["revision"]),
            "grader": GRADER_VERSIONS["gsm8k"],
            "fn": gsm8k_grade,
        },
        bbh.NAME: {
            "revision": bbh.PINNED_COMMIT,
            "grader": GRADER_VERSIONS["bbh"],
            "fn": bbh_grade,
        },
    }


def grader_sha256(root: Path | None = None) -> str:
    base = Path(root) if root is not None else PROJECT_ROOT
    return hashlib.sha256((base / GRADER_SOURCE).read_bytes()).hexdigest()


def _row_problems(r: Mapping[str, Any], gold: Mapping[str, str]) -> list[str]:
    out = []
    if not isinstance(r.get("trace"), str):
        out.append("trace is not a string")
    if not isinstance(r.get("truncated"), bool):
        out.append(f"truncated is not a boolean: {r.get('truncated')!r}")
    if str(r.get("pid")) not in gold:
        out.append(f"no gold answer for {r.get('pid')!r}")
    return out


def rescore(
    rows: Iterable[Mapping[str, Any]],
    gold: Mapping[str, str],
    *,
    bench: str,
    revision: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Primary scores for every row, or INVALID with the reasons."""
    if bench == "math500":
        raise ValueError("MATH-500 keeps its frozen in-row grader; not re-scored")
    reg = registry(root)
    if bench not in reg:
        raise ValueError(f"no frozen grader for benchmark {bench!r}")
    entry = reg[bench]
    if revision != entry["revision"]:
        raise ValueError(
            f"{bench} revision {revision!r} is not the frozen {entry['revision']!r}"
        )
    fn: Callable[..., tuple[bool, str]] = entry["fn"]
    gsha = grader_sha256(root)
    scores, problems = [], {}
    for i, r in enumerate(rows):
        probs = _row_problems(r, gold)
        if probs:
            problems[f"row {i}"] = probs
            continue
        correct, rule = fn(
            r["trace"], str(gold[str(r["pid"])]), truncated=r["truncated"]
        )
        scores.append(
            {
                "pid": str(r["pid"]),
                "press": r.get("press"),
                "budget": r.get("budget"),
                "seed": (r.get("decoding") or {}).get("seed"),
                "correct": bool(correct),
                "rule": rule,
                "truncated": r["truncated"],
                "trace_sha256": hashlib.sha256(r["trace"].encode()).hexdigest(),
                "in_row_correct": r.get("correct"),
            }
        )
    return {
        "status": INVALID if problems else FINAL,
        "bench": bench,
        "revision": revision,
        "grader": entry["grader"],
        "grader_sha256": gsha,
        "n_rows": len(scores) + len(problems),
        "problems": dict(list(problems.items())[:20]),
        "scores": scores if not problems else [],
    }

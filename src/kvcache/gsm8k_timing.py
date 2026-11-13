"""Read of the GSM8K full-cache timing pilot (docs/paper/GSM8K_TIMING_PROPOSAL.md;
built 2026-11-10, before any GSM8K generation exists).

Scientific rationale. The pilot exists to price the matched GSM8K replication,
so its read must be exact about WHICH conditions it describes and HOW they are
scored:

* Coverage. The design is the frozen calibration split x the design seeds
  (24 x 2 = 48 keys). The read is FINAL only when every expected key has
  exactly one result row and exactly one token-ID sidecar, the two agree
  (identity, seed, manifest split, trace hash, trace token count), no key
  outside the design is present, no line is torn, and no publication envelope
  is still pending. Anything else refuses: a partial pilot is not priced.
* Scoring. The generation row's ``correct`` field comes from the MATH grader
  inside generation, which ``kvcache.task_graders`` documents as a systematic
  false negative on GSM8K answers. The GSM8K primary is the registered
  read-time scorer ``kvcache.task_scoring.rescore`` (the frozen ``gsm8k_grade``
  dispatched by benchmark and pinned revision, recording the grader source
  hash and each trace hash), applied to the STORED trace against the pinned
  gold, with truncation scored incorrect. The row field is reported only as a
  disagreement count, never as a score.
* Timing. Seconds per generated token pools wall clock over generated tokens
  (the same estimator as the MATH-500 bank's 0.0253 s/token). Whether a
  budget would bind is read from the sidecar's prompt token count plus the
  trace token count, against B = 256 and B = 1024.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from kvcache.m5_publish import JOURNAL, QUARANTINE, row_key
from kvcache.task_scoring import FINAL, rescore
from kvcache.token_ids import text_sha256

BUDGETS = (256, 1024)
BBH = "bbh_lde5"  # kvcache.bbh.NAME, the registered BBH scorer key
SCHEMA = "gsm8k-timing-read-v1"

Key = tuple[str, str, str, int]


def expected_keys(pids: Iterable[str], seeds: Iterable[int]) -> set[Key]:
    return {("fullcache", "full", str(p), int(s)) for p in pids for s in seeds}


def _records(base: Path) -> tuple[dict[Key, list[dict[str, Any]]], list[str]]:
    """Every record under ``base`` by key, plus torn-line locations."""
    by_key: dict[Key, list[dict[str, Any]]] = {}
    torn: list[str] = []
    for path in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = row_key(rec)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                torn.append(f"{path.name}:{n}")
                continue
            by_key.setdefault(key, []).append(rec)
    return by_key, torn


def _pending_envelopes(runs_root: Path) -> list[str]:
    jdir = runs_root / JOURNAL
    if not jdir.exists():
        return []
    return sorted(
        p.name for p in jdir.rglob("*.json") if not p.name.endswith(".published.json")
    )


def _quarantined(runs_root: Path) -> int:
    qdir = runs_root / QUARANTINE
    if not qdir.exists():
        return 0
    return sum(
        1
        for p in qdir.rglob("*.jsonl")
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    )


def pair_problems(
    row: Mapping[str, Any], side: Mapping[str, Any], manifest_split: Mapping[str, str]
) -> list[str]:
    """Why a row and its sidecar are not one valid pair (empty: they are)."""
    out = []
    for field in ("bench", "pid", "model", "press", "budget"):
        if row.get(field) != side.get(field):
            out.append(f"{field} differs between row and sidecar")
    if int(row["decoding"]["seed"]) != int(side.get("seed", -1)):
        out.append("seed differs between row and sidecar")
    for name, rec in (("row", row), ("sidecar", side)):
        if dict(rec.get("manifest_split") or {}) != dict(manifest_split):
            out.append(f"{name} manifest_split {rec.get('manifest_split')!r}")
    if side.get("trace_text_sha256") != text_sha256(str(row.get("trace", ""))):
        out.append("sidecar trace hash does not match the row's trace")
    if len(side.get("trace_token_ids") or []) != int(row.get("n_tokens", -1)):
        out.append("sidecar trace token count != row n_tokens")
    if not side.get("prompt_token_ids"):
        out.append("sidecar has no prompt token IDs")
    return out


def validate_coverage(
    runs_root: Path,
    *,
    bench: str,
    pids: Iterable[str],
    seeds: Iterable[int],
    manifest_split: Mapping[str, str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The full-cache design's (row, sidecar) pairs in design order."""
    keys = [("fullcache", "full", str(p), int(s)) for s in seeds for p in pids]
    return validate_pairs(
        runs_root, bench=bench, keys=keys, manifest_split=manifest_split
    )


def validate_pairs(
    runs_root: Path,
    *,
    bench: str,
    keys: list[Key],
    manifest_split: Mapping[str, str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """One matching (row, sidecar) pair per expected key, in ``keys`` order,
    or ValueError naming the problems. Shared by every manifest-bank read."""
    runs_root = Path(runs_root)
    want = set(keys)
    if len(want) != len(keys):
        raise ValueError("the expected key list repeats a key")
    rows, torn_r = _records(runs_root / bench)
    sides, torn_s = _records(runs_root / "token_ids" / bench)
    errors: list[str] = []
    if torn_r or torn_s:
        errors.append(f"torn or unreadable lines: {(torn_r + torn_s)[:3]}")
    pending = _pending_envelopes(runs_root)
    if pending:
        errors.append(f"{len(pending)} publication envelopes still pending")
    for label, got in (("result rows", rows), ("sidecars", sides)):
        missing = sorted(want - set(got))
        extra = sorted(set(got) - want)
        dup = sorted(k for k, v in got.items() if len(v) > 1)
        if missing:
            errors.append(f"{len(missing)} expected keys lack {label}: {missing[0]}")
        if extra:
            errors.append(f"{len(extra)} {label} outside the design: {extra[0]}")
        if dup:
            errors.append(f"{len(dup)} keys have repeated {label}: {dup[0]}")
    if errors:
        raise ValueError("; ".join(errors))
    pairs = []
    for key in keys:
        row, side = rows[key][0], sides[key][0]
        probs = pair_problems(row, side, manifest_split)
        if probs:
            raise ValueError(f"{key}: {probs[0]}")
        pairs.append((row, side))
    return pairs


def _gold_problem(bench: str, gold: str) -> str | None:
    """Why a gold cannot be scored by the benchmark's frozen grader, or None."""
    if bench == "gsm8k":
        try:
            Fraction(gold)
        except (ValueError, ZeroDivisionError):
            return "is not a number"
        return None
    if bench == BBH:
        # the pinned file's target form, which kvcache.bbh.parse_bbh enforces
        return None if re.fullmatch(r"\([A-E]\)", gold) else "is not one option"
    return f"has no registered gold check for {bench!r}"


def score_rows(
    rows: list[Mapping[str, Any]],
    golds: Mapping[str, str],
    *,
    revision: str,
    bench: str = "gsm8k",
) -> dict[str, Any]:
    """The registered read-time scores (``task_scoring.rescore``: ``gsm8k_grade``
    or ``bbh_grade`` by benchmark), in row order, after checking every needed
    gold is well-formed for that grader; ValueError otherwise."""
    for pid in {str(r["pid"]) for r in rows}:
        if pid not in golds:
            raise ValueError(f"no gold answer for {pid}")
        why = _gold_problem(bench, str(golds[pid]))
        if why:
            raise ValueError(f"gold for {pid} {why}: {golds[pid]!r}")
    scored = rescore(rows, golds, bench=bench, revision=revision)
    if scored["status"] != FINAL:
        raise ValueError(f"{bench} re-score INVALID: {scored['problems']}")
    return scored


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    golds: Mapping[str, str],
    *,
    revision: str,
    bench: str = "gsm8k",
) -> dict[str, Any]:
    """Accuracy under the registered scorer, truncation, lengths, timing."""
    scored = score_rows([r for r, _ in pairs], golds, revision=revision, bench=bench)
    conditions = []
    for (row, side), score in zip(pairs, scored["scores"], strict=True):
        correct, rule = score["correct"], score["rule"]
        prompt_len = len(side["prompt_token_ids"])
        conditions.append(
            {
                "pid": row["pid"],
                "seed": int(row["decoding"]["seed"]),
                "correct": bool(correct),
                "rule": rule,
                "row_correct_field": bool(row["correct"]),
                "truncated": bool(row["truncated"]),
                "hit_token_cap": bool(row["hit_token_cap"]),
                "n_tokens": int(row["n_tokens"]),
                "prompt_tokens": prompt_len,
                "wallclock_s": float(row["wallclock_s"]),
            }
        )
    n = len(conditions)
    by_pid: dict[str, list[bool]] = {}
    for c in conditions:
        by_pid.setdefault(c["pid"], []).append(c["correct"])
    lengths = [float(c["n_tokens"]) for c in conditions]
    tokens = sum(c["n_tokens"] for c in conditions)
    wall = sum(c["wallclock_s"] for c in conditions)
    rules: dict[str, int] = {}
    for c in conditions:
        rules[c["rule"]] = rules.get(c["rule"], 0) + 1
    return {
        "n_conditions": n,
        "n_problems": len(by_pid),
        "accuracy": sum(c["correct"] for c in conditions) / n,
        "accuracy_equal_problem": statistics.fmean(
            statistics.fmean(v) for v in by_pid.values()
        ),
        "grader": scored["grader"],
        "grader_sha256": scored["grader_sha256"],
        "grader_rules": dict(sorted(rules.items())),
        "row_correct_field_disagreements": sum(
            c["correct"] != c["row_correct_field"] for c in conditions
        ),
        "truncation_rate": sum(c["truncated"] for c in conditions) / n,
        "hit_token_cap_rate": sum(c["hit_token_cap"] for c in conditions) / n,
        "trace_tokens": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "p90": _quantile(lengths, 0.9),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "total": tokens,
        },
        "seconds_per_token": wall / tokens if tokens else None,
        "generation_gpu_hours": wall / 3600,
        "budget_would_bind": {
            str(b): sum(c["prompt_tokens"] + c["n_tokens"] > b for c in conditions) / n
            for b in BUDGETS
        },
        "conditions": conditions,
    }


def read_timing(
    runs_root: Path,
    *,
    bench: str,
    pids: Iterable[str],
    seeds: Iterable[int],
    manifest_split: Mapping[str, str],
    golds: Mapping[str, str],
) -> dict[str, Any]:
    """The FINAL timing read, or ValueError naming the first problem."""
    pids, seeds = list(pids), list(seeds)
    pairs = validate_coverage(
        runs_root, bench=bench, pids=pids, seeds=seeds, manifest_split=manifest_split
    )
    return {
        "schema": SCHEMA,
        "status": "FINAL",
        "bench": bench,
        "manifest_split": dict(manifest_split),
        "expected_keys": len(expected_keys(pids, seeds)),
        "validated_pairs": len(pairs),
        "quarantined_records": _quarantined(Path(runs_root)),
        "score_source": (
            f"{'bbh_grade' if bench == BBH else 'gsm8k_grade'} on the stored "
            "trace; row 'correct' ignored"
        ),
        **summarize(
            pairs, golds, revision=str(manifest_split["revision"]), bench=bench
        ),
    }

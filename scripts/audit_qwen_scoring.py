#!/usr/bin/env python3
"""S.4 bank-scoring audit: every bank row under the frozen and the secondary
grader (execution plan WP2, 2026-10-05; specification in
docs/protocol/QWEN_SCORING_SENSITIVITY.md).

Thin orchestration over ``kvcache.grading_sensitivity``. For one bank:

1. coverage against the expected (problem, seed) design, no duplicates;
2. the frozen primary score recomputed from the raw trace and compared with
   the stored ``correct`` (any mismatch -> INVALID);
3. the secondary score, its rule and transition, the boundary status;
4. a per-row TSV with the trace's sha256 and the trace tail, so every changed
   or ambiguous score can be checked by a person without the server.

The primary score and its gate outcome are reported unchanged; the secondary
is labelled SENSITIVITY throughout. Rows live on the GPU host; run it there.

    python scripts/audit_qwen_scoring.py --runs-root <runs> --bank qwen3_4b
    python scripts/audit_qwen_scoring.py --runs-root <runs> --bank r1_qwen7b
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_m4 import _load_rows

from kvcache import grading_sensitivity as gs
from kvcache.canonicalize import extract_answer
from kvcache.generation import has_answer_marker
from kvcache.grading import is_equiv
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import FINAL, INVALID, exit_code, frozen_pids, retire_stale

OUT = PROJECT_ROOT / "results" / "diag" / "qwen_scoring"
BENCH = "math500"
SEEDS = (11, 23, 47, 89)
BANK_BAND = (0.85, 0.99)
BANKS = {
    # run dir, subset size of the expected design, model subtree
    "qwen3_4b": {"run": "fc-qwen3-4b", "subset": 50},
    "r1_qwen7b": {"run": "math500-fc-powered-merged", "subset": 100},
}
TAIL_CHARS = 400
# Mechanical mapping of each primary-incorrect row to the S.4 note's categories
# (PREREGISTRATION S.4, 2026-10-04), so the 4/2/6/4/11/4 split can be checked.
CATEGORY = {
    "TEXT_WRAPPER": "text wrapper",
    "UNORDERED_LIST": "set order",
    "BASE_SUBSCRIPT": "base subscript",
    "UNCLOSED_THINKING": "unclosed thinking",
    "NO_CLOSED_REASONING": "unclosed thinking",
}


def _problems(n: int) -> dict[str, Any]:
    import yaml

    from kvcache.datasets import load_benchmark

    bcfg = yaml.safe_load((PROJECT_ROOT / "configs/benchmarks.yaml").read_text())
    probs = load_benchmark(BENCH, bcfg["benchmarks"][BENCH], subset=n)
    return {p.pid: p for p in probs}


def _category(r: dict[str, Any]) -> str:
    if r["secondary_extraction"] == "TERMINAL_BOX_BLOCK" and r["secondary_verdict"]:
        return "multi-box"
    return CATEGORY.get(r["secondary_rule"], "wrong value or other")


def audit(runs: Path, bank: str) -> tuple[dict[str, Any], list[list[Any]]]:
    spec = BANKS[bank]
    probs = _problems(spec["subset"])
    expected_pids = set(probs)
    if spec["subset"] == 100 and expected_pids != set(
        frozen_pids(100, root=PROJECT_ROOT)
    ):
        raise SystemExit("REFUSED: the loader's 100 problems differ from grid100")
    expected = {(p, s) for p in expected_pids for s in SEEDS}
    rows = [
        r
        for r in _load_rows(runs / spec["run"], BENCH)
        if (r["pid"], int(r["decoding"]["seed"])) in expected
    ]
    seen = collections.Counter((r["pid"], int(r["decoding"]["seed"])) for r in rows)
    dupes = sorted(k for k, n in seen.items() if n > 1)
    missing = sorted(expected - set(seen))
    out_rows, tsv, mismatches = [], [], []
    for r in sorted(rows, key=lambda r: (r["pid"], int(r["decoding"]["seed"]))):
        key = (r["pid"], int(r["decoding"]["seed"]))
        p = probs[r["pid"]]
        trace = r["trace"]
        has = has_answer_marker(trace)
        recomputed = (
            not (bool(r["hit_token_cap"]) and not has)
            and has
            and is_equiv(p.gold_answer, extract_answer(trace))
        )
        if recomputed != bool(r["correct"]):
            mismatches.append(list(key))
        s = gs.score_row(
            trace=trace,
            gold=p.gold_answer,
            problem_text=p.question,
            hit_cap=bool(r["hit_token_cap"]),
            primary=bool(r["correct"]),
        )
        s["category"] = None if r["correct"] else _category(s)
        out_rows.append(s)
        tsv.append(
            [
                key[0],
                key[1],
                p.gold_answer,
                int(bool(r["correct"])),
                int(recomputed),
                s["boundary_status"],
                s["primary_answer"],
                s["secondary_answer"] if s["secondary_answer"] is not None else "",
                "" if s["secondary_verdict"] is None else int(s["secondary_verdict"]),
                s["secondary_rule"],
                s["transition"],
                s["category"] or "",
                hashlib.sha256(trace.encode("utf-8")).hexdigest(),
                trace[-TAIL_CHARS:].replace("\t", " ").replace("\n", "\\n"),
            ]
        )
    n = len(out_rows)
    primary = sum(s["primary_correct"] for s in out_rows)
    point = sum(s["secondary_point"] for s in out_rows)
    ambiguous = [s for s in out_rows if s["secondary_verdict"] is None]
    amb_primary_correct = sum(s["primary_correct"] for s in ambiguous)
    status = FINAL if not (dupes or missing or mismatches) else INVALID
    acc = (lambda k: k / n) if n else (lambda k: None)
    summary = {
        "bank": bank,
        "run": spec["run"],
        "status": status,
        "coverage": {
            "n_expected": len(expected),
            "n_rows": n,
            "missing": [list(k) for k in missing[:20]],
            "duplicates": [list(k) for k in dupes[:20]],
        },
        "primary": {
            "grader": "hendrycks-is-equiv-v1 on extract_answer(trace) (FROZEN)",
            "correct": primary,
            "accuracy": acc(primary),
            "recompute_mismatches": mismatches,
            "band": list(BANK_BAND),
            "gate": (
                "PASS" if n and BANK_BAND[0] <= primary / n <= BANK_BAND[1] else "FAIL"
            ),
        },
        "secondary_SENSITIVITY": {
            "version": gs.SECONDARY_VERSION,
            "point_correct": point,
            "point_accuracy": acc(point),
            "n_ambiguous": len(ambiguous),
            "bounds_accuracy": [
                acc(point - amb_primary_correct),
                acc(point - amb_primary_correct + len(ambiguous)),
            ],
            "note": "ambiguous rows keep their primary score in the point "
            "estimate; bounds score all of them incorrect / correct. Not a gate "
            "input: the registered gate is read on the primary score only.",
        },
        "transitions": dict(
            collections.Counter(
                f"{s['transition']}:{s['secondary_rule']}" for s in out_rows
            )
        ),
        "boundary_status": dict(
            collections.Counter(s["boundary_status"] for s in out_rows)
        ),
        "primary_incorrect_categories": dict(
            collections.Counter(s["category"] for s in out_rows if s["category"])
        ),
    }
    return summary, tsv


def _diff(old_tsv: Path, new_tsv: Path) -> dict[str, Any]:
    """Rows whose secondary verdict or rule differ between two versions."""

    def load(p: Path) -> dict[tuple[str, str], dict[str, str]]:
        with p.open(encoding="utf-8") as fh:
            return {
                (r["pid"], r["seed"]): r for r in csv.DictReader(fh, delimiter="\t")
            }

    old, new = load(old_tsv), load(new_tsv)
    if set(old) != set(new):
        raise SystemExit("REFUSED: the previous table covers different rows")
    fields = ("secondary_verdict", "secondary_rule", "transition")
    changed = [
        {
            "pid": k[0],
            "seed": k[1],
            **{f"{f}_old": old[k][f] for f in fields},
            **{f"{f}_new": new[k][f] for f in fields},
        }
        for k in sorted(old)
        if any(old[k][f] != new[k][f] for f in fields)
    ]
    return {"previous": old_tsv.name, "n_changed": len(changed), "rows": changed}


def main() -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--bank", choices=sorted(BANKS), required=True)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="an earlier version's conditions.tsv: record every row whose "
        "secondary verdict or rule changed (version-to-version transition diff)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OUT = a.out_dir
    if a.dry_run:
        print(json.dumps({"command": "audit_qwen_scoring", "bank": a.bank}))
        return 0
    name = f"bank_audit_{a.bank}"
    OUT.mkdir(parents=True, exist_ok=True)
    retired = retire_stale(OUT / f"{name}.json")
    retire_stale(OUT / f"{name}.conditions.tsv")
    try:
        summary, tsv = audit(a.runs_root, a.bank)
    except (SystemExit, Exception) as exc:  # noqa: BLE001 -- evidence, then exit 2
        if isinstance(exc, SystemExit) and exc.code in (0, None):
            raise
        msg = str(exc.code) if isinstance(exc, SystemExit) else repr(exc)
        summary, tsv = {"bank": a.bank, "status": INVALID, "error": msg}, []
    if tsv:
        path = OUT / f"{name}.conditions.tsv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(
                [
                    "pid",
                    "seed",
                    "gold",
                    "primary_stored",
                    "primary_recomputed",
                    "boundary_status",
                    "primary_answer",
                    "secondary_answer",
                    "secondary_verdict",
                    "secondary_rule",
                    "transition",
                    "category",
                    "trace_sha256",
                    f"trace_tail_{TAIL_CHARS}",
                ]
            )
            w.writerows(tsv)
        if a.previous is not None:
            summary["changes_vs_previous"] = _diff(a.previous, path)
        summary["sufficient_statistics_tsv"] = path.name
        summary["sufficient_statistics_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    summary = {
        "reproduce": "python scripts/audit_qwen_scoring.py " + " ".join(sys.argv[1:]),
        "retired_previous_output": retired.name if retired else None,
        **summary,
    }
    write_json_atomic(OUT / f"{name}.json", summary)
    print(f"wrote {OUT / name}.json [{summary['status']}]")
    return exit_code(summary["status"], partial_ok=False)


if __name__ == "__main__":
    raise SystemExit(main())

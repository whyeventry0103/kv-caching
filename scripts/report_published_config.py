#!/usr/bin/env python3
"""Read the R-KV published-configuration cell against its own reference bank.

PREREGISTRATION Amendment N.2: R1-Distill-Llama-8B on AIME-24 at budget 2048,
the model, benchmark and cache ratio of R-KV's 105%-at-16% headline. The read is
an accuracy ratio plus a problem-clustered CI on the difference; both the
statistic (``kvcache.stats.paired_policy_contrast``) and the branch rule
(``kvcache.stats.published_config_branch``) live in the package, so the verdict
is a function of the artifact rather than of whoever writes the paragraph.

Amendment K: the two accuracies are computed over the SAME (problem, seed)
population, and the script refuses rather than silently comparing a cell against
an unmatched bank -- that mismatch is the exact defect Amendment K exists for.

    python scripts/report_published_config.py \
        --fullcache-root <runs>/fc-aime24-llama8b-cap32768 \
        --cell-root <runs>/rkvpub-g0 <runs>/rkvpub-g1 \
        --bench aime24 --out rkv_published.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.stats import paired_policy_contrast, published_config_branch

Key = tuple[str, int]


def load_rows(roots: list[Path], bench: str) -> list[dict[str, Any]]:
    """Every non-token-id row under ``roots``/<bench>, deduplicated on its key.

    Seed-sharded runs are passed together; a (pid, seed) appearing twice across
    shards is a launcher bug, so it raises instead of being averaged away.
    """
    rows: dict[Key, dict[str, Any]] = {}
    for root in roots:
        for path in sorted(
            glob.glob(str(root / bench / "**" / "*.jsonl"), recursive=True)
        ):
            if "token_ids" in Path(path).parts:
                continue
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row["pid"], int(row["decoding"]["seed"]))
                    if key in rows:
                        raise SystemExit(f"duplicate condition {key} across {roots}")
                    rows[key] = row
    return list(rows.values())


def _describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ratios = [
        float(r["compression_ratio"])
        for r in rows
        if r.get("compression_ratio") is not None
    ]
    # Truncations stay in the denominator, scored incorrect. The non-truncated
    # accuracy is reported ALONGSIDE, never instead: the two arms here truncate
    # at very different rates, so a reader needs both to see whether a ratio is
    # carried by the cap or by the policy.
    kept = [r for r in rows if not r["truncated"]]
    return {
        "rows": n,
        "correct": sum(bool(r["correct"]) for r in rows),
        "accuracy": sum(bool(r["correct"]) for r in rows) / n,
        "truncation": sum(bool(r["truncated"]) for r in rows) / n,
        "cap_hits": sum(bool(r.get("hit_token_cap")) for r in rows),
        "n_non_truncated": len(kept),
        "accuracy_non_truncated": (
            sum(bool(r["correct"]) for r in kept) / len(kept) if kept else None
        ),
        "mean_tokens": statistics.fmean(r["n_tokens"] for r in rows),
        "realized_compression_ratio": statistics.fmean(ratios) if ratios else None,
    }


def _by_problem(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = collections.defaultdict(list)
    for row in rows:
        out[row["pid"]].append(bool(row["correct"]))
    return dict(out)


def _agreement(
    cell: list[dict[str, Any]], other: list[dict[str, Any]]
) -> dict[str, Any]:
    """Row-level agreement with a second, independent run of the same config.

    A concurrent duplicate launch is normally waste. Where it overlaps the cell
    it is also a free replication check, so it is reported rather than deleted.
    """
    a = {(r["pid"], int(r["decoding"]["seed"])): r for r in cell}
    b = {(r["pid"], int(r["decoding"]["seed"])): r for r in other}
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"overlapping_conditions": 0}
    same_correct = sum(
        1 for k in shared if bool(a[k]["correct"]) == bool(b[k]["correct"])
    )
    same_answer = sum(1 for k in shared if a[k]["answer_canon"] == b[k]["answer_canon"])
    same_tokens = sum(1 for k in shared if a[k]["n_tokens"] == b[k]["n_tokens"])
    return {
        "overlapping_conditions": len(shared),
        "same_correct": same_correct,
        "same_answer_canon": same_answer,
        "same_n_tokens": same_tokens,
        "correct_agreement_rate": same_correct / len(shared),
        "same_trace": sum(1 for k in shared if a[k]["trace"] == b[k]["trace"]),
        "note": (
            "Two independent processes, different run directories, different "
            "timestamps, same pinned seeds. same_trace equal to the overlap "
            "means decoding reproduced exactly; anything less is a determinism "
            "finding and must be reported as one, not averaged away."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcache-root", required=True, type=Path)
    parser.add_argument("--cell-root", required=True, type=Path, nargs="+")
    parser.add_argument(
        "--replicate-root",
        type=Path,
        nargs="*",
        default=[],
        help="optional second run of the same config; reported as an agreement "
        "check on the overlapping conditions, never merged into the cell",
    )
    parser.add_argument("--bench", default="aime24")
    parser.add_argument("--label", default="rkv-published-config")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    fc_rows = load_rows([args.fullcache_root], args.bench)
    cell_rows = load_rows(args.cell_root, args.bench)
    if not fc_rows or not cell_rows:
        raise SystemExit("no rows found; check --fullcache-root/--cell-root")

    fc_keys = {(r["pid"], int(r["decoding"]["seed"])) for r in fc_rows}
    cell_keys = {(r["pid"], int(r["decoding"]["seed"])) for r in cell_rows}
    if fc_keys != cell_keys:
        raise SystemExit(
            "Amendment K: populations differ -- "
            f"{len(cell_keys - fc_keys)} conditions only in the cell, "
            f"{len(fc_keys - cell_keys)} only in the bank"
        )

    fc = _describe(fc_rows)
    cell = _describe(cell_rows)
    ratio = cell["accuracy"] / fc["accuracy"]
    contrast = paired_policy_contrast(_by_problem(cell_rows), _by_problem(fc_rows))
    verdict = published_config_branch(ratio, contrast["ci95"])

    summary: dict[str, Any] = {
        "label": args.label,
        "benchmark": args.bench,
        "reference_convention": "amendment-k-cell-population",
        "n_conditions": len(cell_keys),
        "full_cache": fc,
        "cell": cell,
        "accuracy_ratio_r": ratio,
        "contrast_cell_minus_fullcache": contrast,
        "preregistered_read": verdict,
    }
    if args.replicate_root:
        summary["independent_rerun_agreement"] = _agreement(
            cell_rows, load_rows(list(args.replicate_root), args.bench)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

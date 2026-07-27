#!/usr/bin/env python3
"""Emit prompt lengths and realized compression ratios as an artifact.

The paper quotes "budget 256 is 3.8% of the mean 6,725-token trace" and
"budget 2048 is 56.6% of its mean 3,618-token trace". Neither was reproducible,
because the denominator is prompt + generation and the prompt component (a few
dozen tokens) existed in no artifact -- only the generation mean did.

Prompt lengths are read from the token-ID sidecars rather than re-tokenized.
Those records store the exact ``prompt_token_ids`` used at generation time and
are written through ``validate_token_ids_record``, which enforces a
decode-encode round-trip -- so they are ground truth, and using them avoids
loading a tokenizer (and avoids any risk of re-rendering the prompt by a
slightly different path than the one that produced the runs).

    python scripts/report_prompt_lengths.py --runs-root <runs> \
        --run grid-diss256-tok --run grid-diss2048-tok --out prompt_lengths.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.compute_budget import realized_compression


def collect(runs_root: Path, runs: list[str]) -> dict[tuple[str, str], list[int]]:
    """Prompt token counts per (press, budget), from token-ID sidecars."""
    lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    for name in runs:
        pattern = str(runs_root / name / "token_ids" / "**" / "*.jsonl")
        paths = glob.glob(pattern, recursive=True)
        if not paths:
            raise SystemExit(f"no token_ids sidecars under {runs_root / name}")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    key = (str(record["press"]), str(record["budget"]))
                    lengths[key].append(len(record["prompt_token_ids"]))
    return dict(lengths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, dest="runs")
    parser.add_argument(
        "--generated-mean",
        action="append",
        default=[],
        metavar="PRESS:BUDGET:MEAN",
        help="mean generated length for a cell, to realize its ratio",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    lengths = collect(args.runs_root, args.runs)
    by_cell: dict[str, Any] = {}
    for (press, budget), values in sorted(lengths.items()):
        by_cell[f"{press}-{budget}"] = {
            "press": press,
            "budget": budget,
            "n": len(values),
            "mean_prompt_tokens": statistics.fmean(values),
            "median_prompt_tokens": statistics.median(values),
            "min_prompt_tokens": min(values),
            "max_prompt_tokens": max(values),
        }

    pooled = [v for values in lengths.values() for v in values]
    summary: dict[str, Any] = {
        "by_cell": by_cell,
        "pooled": {
            "n": len(pooled),
            "mean_prompt_tokens": statistics.fmean(pooled),
            "median_prompt_tokens": statistics.median(pooled),
        },
        "runs": list(args.runs),
        "note": (
            "Prompt lengths from token-ID sidecars (exact generation-time "
            "tokens, decode-encode validated), not re-tokenized. Realized "
            "ratios use budget / (prompt + generation); the "
            "generation-only figure is reported alongside to show the "
            "difference the prompt term makes."
        ),
    }

    ratios = {}
    for spec in args.generated_mean:
        press, budget, mean = spec.split(":")
        cell = by_cell.get(f"{press}-{budget}")
        prompt_mean = (
            cell["mean_prompt_tokens"]
            if cell
            else summary["pooled"]["mean_prompt_tokens"]
        )
        ratios[f"{press}-{budget}"] = realized_compression(
            int(budget), prompt_mean, float(mean)
        )
    if ratios:
        summary["realized"] = ratios

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

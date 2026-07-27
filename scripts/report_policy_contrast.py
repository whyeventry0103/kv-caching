#!/usr/bin/env python3
"""Emit the paired policy contrast (random-beats-recency) as an artifact.

Contribution bullet 3 and §4.1 quote a mean difference, a problem-clustered CI,
and a per-problem win/tie/loss breakdown. None of the three had a machine-
readable source. Statistic lives in ``kvcache.stats.paired_policy_contrast``.

    python scripts/report_policy_contrast.py --grid-root <dir> \
        --press-a random --press-b streamingllm --budget 256 --out contrast.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.stats import paired_policy_contrast


def load_rows(root: Path, bench: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(str(root / bench / "**" / "*.jsonl"), recursive=True)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", required=True, type=Path)
    parser.add_argument("--press-a", required=True)
    parser.add_argument("--press-b", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--bench", default="math500")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = load_rows(args.grid_root, args.bench)
    if not rows:
        raise SystemExit(f"no rows under {args.grid_root}/{args.bench}")

    by_press: dict[str, dict[str, list[bool]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in rows:
        if str(row["budget"]) != str(args.budget):
            continue
        by_press[row["press"]][row["pid"]].append(bool(row["correct"]))

    for press in (args.press_a, args.press_b):
        if press not in by_press:
            raise SystemExit(
                f"press {press!r} absent at budget {args.budget}; "
                f"present: {sorted(by_press)}"
            )

    summary = paired_policy_contrast(by_press[args.press_a], by_press[args.press_b])
    summary["press_a"] = args.press_a
    summary["press_b"] = args.press_b
    summary["budget"] = str(args.budget)
    summary["benchmark"] = args.bench
    summary["accuracy_a"] = sum(sum(v) for v in by_press[args.press_a].values()) / sum(
        len(v) for v in by_press[args.press_a].values()
    )
    summary["accuracy_b"] = sum(sum(v) for v in by_press[args.press_b].values()) / sum(
        len(v) for v in by_press[args.press_b].values()
    )
    summary["note"] = (
        "Report as 'non-negative on N/M problems (W strict wins / T ties / "
        "L losses)', never as 'beats on N/M' -- PAPER_FINDINGS section 5."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Task 3 / Amendment P.3: does AUC divergence predict the accuracy residual?

A CROSS-SCALE test -- a TV-scale quantity predicting an accuracy-scale one.
Legitimate only as a prediction that can fail; never as an assertion that
divergence explains accuracy.

Reported POOLED and WITHIN-BUDGET. Both AUC and the residual scale with trace
length and with budget, so a pooled cross-cell correlation can be a budget
effect wearing a mechanism's clothes. If it holds only pooled, P.3's
BUDGET EFFECT ONLY branch applies.

    python scripts/report_auc_residual.py --runs-root <runs> \\
        --profiles replay-ext-diss256 --measure tv --out auc_residual.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.stats import correlation_with_cluster_bootstrap  # noqa: E402

CELL = re.compile(r"^(?P<press>[a-z0-9]+)-(?P<budget>\d+)-s(?P<seed>\d+)$")
# Per-cell residuals, keyed (press, budget) -> catalogue entry in paper_numbers.
RESIDUAL_KEYS = {
    ("random", "256"): "diss256.random.residual",
    ("streamingllm", "256"): "diss256.streamingllm.residual",
    ("h2o", "256"): "h2o256.h2o.residual",
    ("rkv", "256"): "rkv256.rkv.residual",
    ("random", "2048"): "b2048.random.residual",
    ("streamingllm", "2048"): "b2048.streamingllm.residual",
    ("h2o", "2048"): "b2048h2o.h2o.residual",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--profiles", required=True, help="run name holding v2 profiles")
    p.add_argument("--measure", default="tv", choices=["tv", "jsd"])
    p.add_argument(
        "--catalogue", type=Path, default=PROJECT_ROOT / "paper_numbers.json"
    )
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()

    catalogue = {
        e["key"]: float(e["value"])
        for e in json.loads(a.catalogue.read_text(encoding="utf-8"))["entries"]
    }

    # Per-problem AUC per cell. The problem is the clustering unit, so several
    # seeds of one problem collapse into that problem's mean.
    per_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    versions: set[str] = set()
    root = a.runs_root / a.profiles / "profiles"
    for path in sorted(root.rglob("*.json")):
        m = CELL.match(path.parent.name)
        if not m:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions.add(str(payload.get("schema_version")))
        series = payload["estimate"] if a.measure == "tv" else payload.get("jsd")
        if not series:
            continue
        problem = str(payload["reference_key"][1])
        per_cell[(m["press"], m["budget"])][problem].append(
            sum(float(v) for v in series) / len(series)
        )

    if a.measure == "jsd" and versions and versions != {"tv-profile-v2"}:
        raise SystemExit(f"JSD needs v2 profiles only; found {sorted(versions)}")

    cells: dict[str, dict[str, dict[str, float]]] = {}
    skipped: list[str] = []
    for (press, budget), problems in sorted(per_cell.items()):
        key = RESIDUAL_KEYS.get((press, budget))
        if key is None or key not in catalogue:
            skipped.append(f"{press}-{budget}: no catalogued residual")
            continue
        residual = catalogue[key]
        auc = {q: sum(v) / len(v) for q, v in problems.items()}
        # The residual is a CELL-level quantity: it is held constant across that
        # cell's problems rather than invented per problem. The bootstrap
        # therefore resamples problems for AUC only, which is the honest thing
        # to do and is stated in the artifact rather than hidden.
        cells[f"{press}-{budget}"] = {
            "auc": auc,
            "residual": dict.fromkeys(auc, residual),
        }

    payload: dict[str, Any] = {
        "amendment": "P.3",
        "measure": a.measure,
        "schema_versions": sorted(versions),
        "n_cells": len(cells),
        "skipped": skipped,
        "per_cell_summary": {
            name: {
                "auc_mean": sum(c["auc"].values()) / len(c["auc"]),
                "residual": next(iter(c["residual"].values())),
                "n_problems": len(c["auc"]),
            }
            for name, c in cells.items()
        },
        "residual_is_cell_level": (
            "The residual is constant within a cell, so the clustered bootstrap "
            "carries AUC's sampling error only. It is NOT a joint interval over "
            "both quantities."
        ),
    }

    if len(cells) >= 3:
        payload["pooled"] = correlation_with_cluster_bootstrap(cells)
    else:
        payload["pooled"] = {"skipped": f"only {len(cells)} cells; not a correlation"}

    by_budget: dict[str, dict] = defaultdict(dict)
    for name, c in cells.items():
        by_budget[name.rsplit("-", 1)[1]][name] = c
    payload["within_budget"] = {}
    for budget, group in sorted(by_budget.items()):
        if len(group) >= 3:
            payload["within_budget"][budget] = correlation_with_cluster_bootstrap(group)
        else:
            payload["within_budget"][budget] = {
                "skipped": f"only {len(group)} cells at budget {budget}"
            }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

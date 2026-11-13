#!/usr/bin/env python3
"""Audit of the EXISTING AIME-24 evidence for the M5-B supervisor report
(2026-11-13; read-only over finished rows, GPU host, CPU only).

The AIME-24 cells (Amendments N/O; read 2026-07-18) are a smaller and
differently configured comparison than the new four-policy replication:
Qwen-7B, 30 problems x seeds 11/23/47/89, budget 2048, prompt EVICTABLE only,
three policies (Random, StreamingLLM, R-KV; no H2O-style), cap 16384. They are
re-read here through the same frozen per-dataset read (``kvcache.m5b_reads``)
so the report can show them beside GSM8K and BBH, labelled as prior evidence,
never as the new replication. AIME's registered score is the in-row MATH
grader (integer answers). Each cell must cover the 120 keys exactly.

    python scripts/audit_aime_m5b.py --dry-run
    python scripts/audit_aime_m5b.py \\
        --out results/diag/meeting5/replication/aime24_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.m5b_reads import read_dataset
from kvcache.meeting5_inventory import load_rows

RUNS = Path("/home/sk10741/kvcache-storage/runs")
MODEL = "r1_distill_qwen_7b"
SEEDS = (11, 23, 47, 89)
PRESSES = ("random", "streamingllm", "rkv")
BUDGET = 2048


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, default=RUNS)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def _seed(r: dict[str, Any]) -> int:
    return int((r.get("decoding") or {}).get("seed", r.get("seed")))


def _one(rows: list[dict], label: str) -> dict:
    by: dict[tuple, list] = {}
    for r in rows:
        by.setdefault((str(r["pid"]), _seed(r)), []).append(r)
    dup = [k for k, v in by.items() if len(v) > 1]
    if dup:
        raise ValueError(f"{label}: repeated keys {dup[:2]}")
    return {k: v[0] for k, v in by.items()}


def main() -> int:
    import yaml

    args = _parser().parse_args()
    if args.dry_run:
        print(json.dumps({"presses": PRESSES, "seeds": SEEDS, "budget": BUDGET}))
        print("DRY RUN: nothing read")
        return 0
    root = args.runs_root
    fc_all = [
        r
        for r in load_rows(root / "fc-aime24")
        if r.get("model") == MODEL and _seed(r) in SEEDS
    ]
    fc_rows = _one(fc_all, "fc")
    expected = set(fc_rows)
    if len(expected) != 120:
        raise SystemExit(f"REFUSED: full-cache bank covers {len(expected)} != 120")
    grid = [r for r in load_rows(root / "grid-aime24-2048") if r.get("model") == MODEL]
    sides = {}
    for path in sorted((root / "grid-aime24-2048" / "token_ids").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s = json.loads(line)
                sides[(str(s["press"]), str(s["pid"]), _seed(s))] = len(
                    s["prompt_token_ids"]
                )
    arms, cells = {}, {}
    for press in PRESSES:
        rows = _one([r for r in grid if r["press"] == press], press)
        if set(rows) != expected:
            cells[press] = "INVALID: coverage differs from the 120 keys"
            continue
        arms[(press, False)] = {
            k: {
                "correct": bool(r["correct"]),
                "truncated": bool(r["truncated"]),
                "compression_ratio": float(r["compression_ratio"]),
                "prompt_tokens": sides[(press, k[0], k[1])],
                "n_tokens": int(r["n_tokens"]),
                "n_prunes": None,
            }
            for k, r in rows.items()
        }
        cells[press] = "FINAL"
    readouts: dict = {}
    for rec in load_rows(root / "readout-aime24-2048"):
        ref, press = rec["reference_key"], str(rec["condition"]["press"])
        key = (str(ref[1]), int(ref[5]))
        readouts.setdefault((press, False), {})[key] = {
            "defined": rec.get("readout_defined") is True,
            "correct": (
                bool(rec.get("readout_correct"))
                if rec.get("readout_defined") is True
                else None
            ),
        }
    for a in list(readouts):
        if set(readouts[a]) != expected:
            cells[f"readout-{a[0]}"] = "INVALID: coverage differs"
            del readouts[a]
    fc = {
        k: {
            "correct": bool(r["correct"]),
            "truncated": bool(r["truncated"]),
            "n_tokens": int(r["n_tokens"]),
        }
        for k, r in fc_rows.items()
    }
    spec = yaml.safe_load(
        (PROJECT_ROOT / "configs/meeting5/m5a_read.yaml").read_text(encoding="utf-8")
    )
    read = read_dataset(
        fc,
        arms,
        readouts,
        expected=expected,
        budget=BUDGET,
        reference="random",
        pairings=("streamingllm",),
        descriptive=("rkv",),
        equivalence=spec["equivalence"],
    )
    out = {
        "schema": "m5b-aime-audit-v1",
        "label": (
            "PRIOR EVIDENCE, differently configured: AIME-24, Qwen-7B, 30 problems "
            "x 4 seeds, B=2048, prompt evictable only, Random / StreamingLLM / "
            "R-KV (no H2O-style). Not the new four-policy replication; no "
            "protection arm, so no protection effect is estimable."
        ),
        "cells": cells,
        "read": read,
    }
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    side = read["sides"]["unprotected"]
    print(json.dumps({k: side[k] for k in ("contrasts",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

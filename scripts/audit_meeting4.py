#!/usr/bin/env python3
"""Meeting #4 audit (2026-10-02): thin orchestration over the package reads.

Runs where the rows live (the GPU host); needs no GPU. Every subcommand writes
one JSON under ``docs/results/meeting4_evidence/`` (``--out-dir`` to redirect).

    python scripts/audit_meeting4.py --runs-root <runs> accuracies
    python scripts/audit_meeting4.py --runs-root <runs> readouts
    python scripts/audit_meeting4.py --runs-root <runs> by-length
    python scripts/audit_meeting4.py --runs-root <runs> paired
    python scripts/audit_meeting4.py --runs-root <runs> matched-keys
    python scripts/audit_meeting4.py retention [--device cuda]

``accuracies`` and ``readouts`` re-score every row from its raw trace with the
package grader against freshly loaded gold answers; they establish consistency
with that grader, not independent answer semantics.
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_m4 import _jsonl, _load_rows, _run_key

from kvcache.canonicalize import extract_answer
from kvcache.generation import has_answer_marker
from kvcache.grading import is_equiv
from kvcache.metrics import audit_reads, retention
from kvcache.provenance import write_json_atomic
from kvcache.read_validation import frozen_pids

OUT = PROJECT_ROOT / "docs" / "results" / "meeting4_evidence"
FC_RUN = "math500-fc-powered-merged"
BENCH = "math500"
GRID_RUNS = {
    "random": "grid-diss256-tok",
    "streamingllm": "grid-diss256-tok",
    "h2o": "grid-diss256-h2o",
    "rkv": "grid-diss256-rkv",
}
READOUT_RUNS = [
    "readout-diss256-g0",
    "readout-diss256-g1",
    "readout-diss256-h2o",
    "readout-diss256-rkv",
]
REVERSE_RUNS = [
    "reverse-readout-diss256",
    "reverse-readout-diss256-h2o",
    "reverse-readout-diss256-rkv",
]
SEEDS = (11, 23, 47, 89)


def _gold(runs_root: Path) -> dict[str, str]:
    import yaml

    from kvcache.datasets import load_benchmark

    bcfg = yaml.safe_load((PROJECT_ROOT / "configs/benchmarks.yaml").read_text())
    probs = load_benchmark(
        BENCH, bcfg["benchmarks"][BENCH], subset=100, allow_download=False
    )
    return {p.pid: p.gold_answer for p in probs}


def _rescore(trace: str, hit_cap: bool, gold: str) -> bool:
    has = has_answer_marker(trace)
    truncated = bool(hit_cap) and not has
    return (not truncated) and has and is_equiv(gold, extract_answer(trace))


def _fc(runs: Path) -> dict[tuple[str, int], dict[str, Any]]:
    pids = set(frozen_pids(100, root=PROJECT_ROOT))
    return {
        _run_key(r): r
        for r in _load_rows(runs / FC_RUN, BENCH)
        if r["pid"] in pids and int(r["decoding"]["seed"]) in SEEDS
    }


def _free(runs: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[str, int], dict[str, Any]]] = collections.defaultdict(
        dict
    )
    for run in sorted(set(GRID_RUNS.values())):
        for r in _load_rows(runs / run, BENCH):
            if str(r["budget"]) == "256":
                out[r["press"]][_run_key(r)] = r
    return out


def _readout(runs: Path, run_names: list[str], *, reverse: bool) -> dict:
    out: dict[str, dict[tuple[str, int], dict[str, Any]]] = collections.defaultdict(
        dict
    )
    for run in run_names:
        for path in sorted((runs / run / "readout").rglob("*.jsonl")):
            for r in _jsonl(path):
                ref = r["reference_key"]
                press = str(ref[3]) if reverse else str(r["condition"]["press"])
                out[press][(str(ref[1]), int(ref[5]))] = r
    return out


# ------------------------------------------------------------------ reads
def accuracies(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    gold = _gold(runs)
    fc = _fc(runs)
    out: dict[str, Any] = {
        "read": "re-scored free-run accuracies, budget 256 and the dose cells",
        "grader": "kvcache.grading.is_equiv (Hendrycks) on extract_answer of the "
        "raw trace",
        "fc_matched": {
            "n": len(fc),
            "accuracy": statistics.fmean(r["correct"] for r in fc.values()),
        },
        "cells": [],
    }
    for run in [
        "grid-diss256-tok",
        "grid-diss256-h2o",
        "grid-diss256-rkv",
        "grid-diss1024",
        "grid-diss2048-tok",
    ]:
        by: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for r in _load_rows(runs / run, BENCH):
            by[(r["press"], str(r["budget"]))].append(r)
        for (press, b), v in sorted(by.items()):
            keys = [_run_key(r) for r in v]
            res = [_rescore(r["trace"], r["hit_token_cap"], gold[r["pid"]]) for r in v]
            out["cells"].append(
                {
                    "run": run,
                    "press": press,
                    "budget": b,
                    "n": len(v),
                    "duplicates": len(keys) - len(set(keys)),
                    "acc_stored": statistics.fmean(bool(r["correct"]) for r in v),
                    "acc_rescored": statistics.fmean(res),
                    "mismatches": sum(
                        1 for r, c in zip(v, res, strict=True) if c != r["correct"]
                    ),
                    "truncation": statistics.fmean(bool(r["truncated"]) for r in v),
                    "fc_on_same_keys": statistics.fmean(fc[k]["correct"] for k in keys),
                    "mean_tokens": statistics.fmean(r["n_tokens"] for r in v),
                }
            )
    out["n_rows_rescored"] = sum(c["n"] for c in out["cells"])
    return out


def readouts(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    gold = _gold(runs)
    fc = _fc(runs)
    free = _free(runs)
    out: dict[str, Any] = {
        "read": "re-scored readouts, budget 256",
        "forward": {},
        "reverse": {},
    }
    for label, run_names, reverse in (
        ("forward", READOUT_RUNS, False),
        ("reverse", REVERSE_RUNS, True),
    ):
        for press, rows in _readout(runs, run_names, reverse=reverse).items():
            defined = {
                k: r for k, r in rows.items() if r.get("readout_correct") is not None
            }
            mism = sum(
                1
                for k, r in defined.items()
                if is_equiv(gold[k[0]], extract_answer(r["generated_text"]))
                != r["readout_correct"]
            )
            acc = statistics.fmean(r["readout_correct"] for r in defined.values())
            block = {
                "n_rows": len(rows),
                "n_defined": len(defined),
                "accuracy": acc,
                "rescored_mismatches": mism,
                "fc_on_defined": statistics.fmean(fc[k]["correct"] for k in defined),
            }
            if not reverse:
                comp = statistics.fmean(free[press][k]["correct"] for k in defined)
                block["free_on_defined"] = comp
                block["residual_complete_case"] = comp - acc
            out[label][press] = block
    return out


def by_length(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    fc = _fc(runs)
    free = _free(runs)
    ro = _readout(runs, READOUT_RUNS, reverse=False)
    bins = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
    presses = sorted(free)
    defined_keys = set.intersection(
        *[
            {k for k, r in ro[p].items() if r.get("readout_correct") is not None}
            for p in presses
        ]
    )
    out: dict[str, Any] = {
        "read": "accuracy by full-cache reference length, budget 256",
        "bins": [],
    }
    for lo, hi in bins:
        ks = [k for k in sorted(defined_keys) if lo <= fc[k]["n_tokens"] < hi]
        fks = [
            k
            for k in fc
            if lo <= fc[k]["n_tokens"] < hi and all(k in free[p] for p in presses)
        ]
        if not ks:
            continue
        out["bins"].append(
            {
                "range": [lo, hi],
                "n_readout_defined": len(ks),
                "fc_accuracy": statistics.fmean(fc[k]["correct"] for k in ks),
                "readout": {
                    p: statistics.fmean(ro[p][k]["readout_correct"] for k in ks)
                    for p in presses
                },
                "n_free": len(fks),
                "free": {
                    p: statistics.fmean(free[p][k]["correct"] for k in fks)
                    for p in presses
                },
                "free_mean_tokens": {
                    p: statistics.fmean(free[p][k]["n_tokens"] for k in fks)
                    for p in presses
                },
                "free_truncation": {
                    p: statistics.fmean(free[p][k]["truncated"] for k in fks)
                    for p in presses
                },
            }
        )
    return out


def _tables(runs: Path):
    free = {
        p: {k: bool(r["correct"]) for k, r in rows.items()}
        for p, rows in _free(runs).items()
    }
    ro = {
        p: {
            k: bool(r["readout_correct"])
            for k, r in rows.items()
            if r.get("readout_correct") is not None
        }
        for p, rows in _readout(runs, READOUT_RUNS, reverse=False).items()
    }
    return free, ro


def paired(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    free, ro = _tables(runs)
    pairs = [
        ("random", "streamingllm"),
        ("random", "h2o"),
        ("rkv", "random"),
        ("h2o", "streamingllm"),
    ]
    return {
        "read": "paired policy contrasts (a - b) on the 100 x 4 conditions, budget 256",
        "estimand": "per-condition free, readout (defined set) and residual = "
        "free - readout; equal-problem mean of paired differences, "
        "problem-cluster bootstrap CI",
        "contrasts": audit_reads.paired_policy_contrasts(
            free, ro, pairs, n_resamples=a.n_resamples, seed=a.seed
        ),
    }


def matched_keys(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    free, ro = _tables(runs)
    pids25 = frozen_pids(25, root=PROJECT_ROOT)
    keys25 = {(p, s) for p in pids25 for s in SEEDS}
    cap2048 = _load_rows(runs / "capture-b2048", BENCH)
    elig2048 = {
        (r["pid"], int(r["decoding"]["seed"] if "decoding" in r else r["seed"]))
        for r in cap2048
        if r["press"] == "random" and r.get("n_events", 0) >= 1
    }
    kw = {"n_resamples": a.n_resamples, "seed": a.seed}
    return {
        "read": "accuracy / readout / residual rankings on the capture keys, "
        "budget-256 cells",
        "all_100_problems": audit_reads.rankings_on_keys(
            free, ro, set(free["random"]), **kw
        ),
        "capture_25_problems": audit_reads.rankings_on_keys(free, ro, keys25, **kw),
        "capture_2048_eligible_keys": audit_reads.rankings_on_keys(
            free, ro, elig2048, **kw
        ),
        "note": "the 2048 eligible set is the 21-problem / 69-trace set with a "
        "measured eviction event at budget 2048; accuracies here are still the "
        "budget-256 cells",
    }


def retention_sim(_runs: Path | None, a: argparse.Namespace) -> dict[str, Any]:
    import torch

    kw = dict(
        prompt_len=a.prompt_len,
        budget=256,
        interval=64,
        seed=a.seed,
        n_kv=4,
        device=a.device,
        dtype_name="bfloat16",
    )
    out: dict[str, Any] = {
        "read": "exact slot-dynamics simulation of the presses (no model)",
        "implementation": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": a.device,
            "dtype": "bfloat16",
            "generator_seed": a.seed,
            "reseeded_every_prune": True,
            "note": "the CUDA and CPU generators give different score vectors; the "
            "structure (fixed slot pattern) is the same on both",
        },
        "design": {**kw, "steps": a.steps},
        "policies": {},
    }
    for policy in ("random", "streamingllm"):
        sim = retention.simulate_slots(policy, steps=a.steps, **kw)
        out["policies"][policy] = {
            "summary": retention.retention_summary(
                sim, prompt_len=a.prompt_len, budget=256
            ),
            "cohort_survival_entering_at_1024": retention.cohort_survival(
                policy,
                enter_step=1024,
                cohort=64,
                horizons=[64, 128, 256, 512, 1024, 2048],
                **kw,
            ),
        }
    return out


READS = {
    "accuracies": accuracies,
    "readouts": readouts,
    "by-length": by_length,
    "paired": paired,
    "matched-keys": matched_keys,
    "retention": retention_sim,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt-len", type=int, default=94)
    ap.add_argument("--steps", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("read", choices=sorted(READS))
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "audit_meeting4", "read": a.read}))
        return 0
    if a.read != "retention" and a.runs_root is None:
        raise SystemExit("--runs-root is required for row reads")
    payload = READS[a.read](a.runs_root, a)
    path = a.out_dir / f"{a.read.replace('-', '_')}.json"
    write_json_atomic(
        path,
        {
            "reproduce": "python scripts/audit_meeting4.py " + " ".join(sys.argv[1:]),
            **payload,
        },
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

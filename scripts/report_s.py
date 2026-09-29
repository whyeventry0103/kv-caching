#!/usr/bin/env python3
"""Amendment S reads over rows already on disk (registered 2026-09-30).

Thin orchestration: load rows, call ``kvcache.metrics.s_reads``, write one JSON
per read under ``results/diag/amendment-s/``. Runs where the rows live (the GPU
host) and needs no GPU. A read refuses an incomplete cell unless ``--partial``
is given, and a partial read carries no label.

    python scripts/report_s.py --runs-root <runs> protection --budget 256
    python scripts/report_s.py --runs-root <runs> protection --budget 1024
    python scripts/report_s.py --runs-root <runs> consequence
    python scripts/report_s.py --runs-root <runs> origin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from report_m4 import (
    _condition_from_readout,
    _jsonl,
    _load_rows,
    _reference_key_from_readout,
    _run_key,
)

from kvcache.coverage import equal_problem_bootstrap
from kvcache.metrics import s_reads
from kvcache.presses.prompt_protect import PP_BASES, PP_SUFFIX, effective_budget
from kvcache.provenance import write_json_atomic

OUT = PROJECT_ROOT / "results" / "diag" / "amendment-s"
FC_RUN = "math500-fc-powered-merged"
BENCH = "math500"
CELLS = {
    256: {
        "grid_pp": "grid-pp256",
        "readout_pp": ["readout-pp256"],
        "grid_base": "grid-diss256-tok",
        "readout_base": ["readout-diss256-g0", "readout-diss256-g1"],
    },
    1024: {
        "grid_pp": "grid-pp1024",
        "readout_pp": ["readout-pp1024"],
        "grid_base": "grid-diss1024",
        "readout_base": ["readout-diss1024"],
    },
}
N_CONDITIONS = 400


def _free(root: Path, press: str, budget: int) -> dict[tuple[str, int], bool]:
    out: dict[tuple[str, int], bool] = {}
    for r in _load_rows(root, BENCH):
        if r["press"] == press and str(r["budget"]) == str(budget):
            k = _run_key(r)
            if k in out:
                raise ValueError(f"duplicate free-run row {k} in {root.name}")
            out[k] = bool(r["correct"])
    return out


def _readout(
    roots: list[Path], press: str, budget: int
) -> tuple[dict[tuple[str, int], bool], int]:
    """Defined readout correctness, and the number of readout rows seen."""
    out: dict[tuple[str, int], bool] = {}
    seen = 0
    for root in roots:
        for path in sorted((root / "readout").rglob("*.jsonl")):
            for r in _jsonl(path):
                if _condition_from_readout(r) != (press, str(budget)):
                    continue
                seen += 1
                if r.get("readout_correct") is None:
                    continue
                k = _reference_key_from_readout(r)
                if k in out:
                    raise ValueError(f"duplicate readout row {k} in {root.name}")
                out[k] = bool(r["readout_correct"])
    return out, seen


def _prompt_lengths(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in sorted(root.rglob("*.jsonl")):
        if "token_ids" not in path.parts:
            continue
        for rec in _jsonl(path):
            out[str(rec["pid"])] = len(rec["prompt_token_ids"])
    return out


def _write(name: str, payload: dict[str, Any], argv: list[str]) -> Path:
    payload = {
        "amendment": "S",
        "reproduce": "python scripts/report_s.py " + " ".join(argv),
        **payload,
    }
    path = OUT / f"{name}.json"
    write_json_atomic(path, payload)
    print(f"wrote {path}")
    return path


def protection(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """S.1 for one budget: per press, the residual and the paired contrasts."""
    cell = CELLS[a.budget]
    fc = {_run_key(r): bool(r["correct"]) for r in _load_rows(runs / FC_RUN, BENCH)}
    lengths = _prompt_lengths(runs / cell["grid_base"])
    relaxed = sorted(p for p, n in lengths.items() if effective_budget(a.budget, n)[1])
    out: dict[str, Any] = {
        "read": "S.1 prompt-protected decomposition",
        "budget": a.budget,
        "inputs": cell,
        "budget_relaxed_problems": relaxed,
        "status": "FINAL",
        "per_press": {},
    }
    for base in PP_BASES:
        pp = base + PP_SUFFIX
        free_pp = _free(runs / cell["grid_pp"], pp, a.budget)
        free_base = _free(runs / cell["grid_base"], base, a.budget)
        ro_pp, n_ro_pp = _readout([runs / r for r in cell["readout_pp"]], pp, a.budget)
        ro_base, n_ro_base = _readout(
            [runs / r for r in cell["readout_base"]], base, a.budget
        )
        counts = {
            "free_pp": len(free_pp),
            "free_base": len(free_base),
            "readout_rows_pp": n_ro_pp,
            "readout_rows_base": n_ro_base,
        }
        complete = all(v == N_CONDITIONS for v in counts.values())
        if not complete:
            if not a.partial:
                raise SystemExit(
                    f"REFUSED: {pp} at {a.budget} is incomplete {counts}; never "
                    "read a partial cell into the paper (--partial for a "
                    "labelled progress read)"
                )
            out["status"] = "PARTIAL"
            shared = set(free_pp) & set(free_base)
            free_pp = {k: v for k, v in free_pp.items() if k in shared}
            free_base = {k: v for k, v in free_base.items() if k in shared}
            both = set(ro_pp) & set(ro_base) & shared
            ro_pp = {k: v for k, v in ro_pp.items() if k in both}
            ro_base = {k: v for k, v in ro_base.items() if k in both}
        kw = {"n_resamples": a.n_resamples, "seed": a.seed}
        primary = s_reads.protection_contrasts(free_pp, free_base, ro_pp, ro_base, **kw)
        sensitivity = s_reads.protection_contrasts(
            free_pp, free_base, ro_pp, ro_base, exclude_pids=relaxed, **kw
        )
        if not complete:
            for block in (primary, sensitivity):
                block["residual_label"] = block["protection_effect"] = (
                    "PARTIAL: no label"
                )
        out["per_press"][pp] = {
            "counts": counts,
            "primary": primary,
            "sensitivity_without_budget_relaxed": sensitivity,
            "flip_mass_pp": s_reads.flip_masses(fc, free_pp),
            "flip_mass_base": s_reads.flip_masses(fc, free_base),
        }
    return out


def consequence(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """S.1 paper-level branch, from the two FINAL protection reads."""
    labels: dict[str, dict[int, str]] = {}
    for budget in sorted(CELLS):
        path = OUT / f"protection_b{budget}.json"
        if not path.exists():
            raise SystemExit(f"REFUSED: {path.name} has not been read")
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["status"] != "FINAL":
            raise SystemExit(f"REFUSED: {path.name} is {d['status']}")
        for press, block in d["per_press"].items():
            labels.setdefault(press, {})[budget] = block["primary"]["residual_label"]
    return {
        "read": "S.1 consequence for the paper",
        "residual_labels": labels,
        "consequence": s_reads.paper_consequence(labels),
    }


def origin(runs: Path, a: argparse.Namespace) -> dict[str, Any]:
    """S.2: eligibility, the determinism control, then the ratios."""
    rows = _load_rows(runs / "restore-v3", BENCH)
    v2_rows = _load_rows(runs / "restore-v2", BENCH)
    fc = {_run_key(r): bool(r["correct"]) for r in _load_rows(runs / FC_RUN, BENCH)}
    out: dict[str, Any] = {
        "read": "S.2 restoration by origin",
        "n_rows": len(rows),
        "per_policy": {},
    }
    for press in sorted({r["press"] for r in rows}):
        rs = [r for r in rows if r["press"] == press]
        read = [r for r in rs if r["status"] == "READ"]
        keys = [(r["pid"], int(r["seed"])) for r in read]
        v2 = {
            (r["pid"], int(r["seed"])): {
                "full": r["full_restore"]["window_mean_tv"],
                "none": r["full_restore"]["none_window_mean_tv"],
            }
            for r in v2_rows
            if r["press"] == press
            and r["status"] == "READ"
            and (r.get("full_restore") or {}).get("eligible")
        }
        v3 = {
            (r["pid"], int(r["seed"])): {
                "full": r["arms"]["full"]["window_mean_tv"],
                "none": r["none_window_mean_tv"],
            }
            for r in read
        }
        control = s_reads.determinism_control(v3, v2)
        block: dict[str, Any] = {
            "status_counts": {
                s: sum(1 for r in rs if r["status"] == s)
                for s in sorted({r["status"] for r in rs})
            },
            "eligible": {
                "n_traces": len(keys),
                "n_problems": len({k[0] for k in keys}),
                "fc_accuracy": (
                    equal_problem_bootstrap(
                        {k: float(fc[k]) for k in keys},
                        n_resamples=a.n_resamples,
                        seed=a.seed,
                    )
                    if keys
                    else None
                ),
            },
            "determinism_control": control,
            "pad_max": max(
                (r["arms"]["prompt"]["gate"]["pad_max"] for r in read), default=None
            ),
            "n_prompt_return": sorted({r["n_prompt_return"] for r in read}),
        }
        if control["passed"]:
            tv = {
                arm: {
                    (r["pid"], int(r["seed"])): r["arms"][arm]["window_mean_tv"]
                    for r in read
                }
                for arm in s_reads.ORIGIN_ARMS
            }
            block["ratios"] = s_reads.origin_ratios(
                tv, n_resamples=a.n_resamples, seed=a.seed
            )
        else:
            block["ratios"] = "WITHHELD: the determinism control did not pass"
        out["per_policy"][press] = block
    return out


READS = {"protection": protection, "consequence": consequence, "origin": origin}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--budget", type=int, choices=sorted(CELLS), default=256)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("read", choices=sorted(READS))
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "report_s", "read": a.read, "out": str(OUT)}))
        return 0
    payload = READS[a.read](a.runs_root, a)
    name = f"protection_b{a.budget}" if a.read == "protection" else a.read
    if payload.get("status") == "PARTIAL":
        name += "_partial"
    _write(name, payload, sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

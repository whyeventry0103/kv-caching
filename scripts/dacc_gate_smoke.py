#!/usr/bin/env python3
"""Amendment R.3 real-model gate for onset-gated free-running generation.

The tiny-model tests pin the gate on CPU; this repeats the two decisive checks
on the real model, GPU and stack that the D_acc cells will use, before any of
them may run (queue kind 'marker': the cells `require` this job).

  (a) PARITY. For each policy at budget 256, a gated generation with
      compression_start_step 0 is token-identical to the legacy ungated
      generation (same seed, same press seed), and at least one prune evicted.
  (b) ONSET. A gated generation with compression_start_step 512 at budget 256
      first evicts at forward 575, so its eviction onset is token 576, and it is
      token-identical to the full-cache generation for tokens [0, 576). The
      achieved exposure is n_tokens - 576.

max_new_tokens is short (1024) because the checks resolve by token 576; the
cells themselves run at the family cap 16384. On PASS it writes
<KVCACHE_BASE_RUNS_ROOT>/dacc-gate-smoke-PASS.json; on FAIL it writes the
FAIL report next to it and exits 1, which blocks every D_acc cell.

    python scripts/dacc_gate_smoke.py \
        --config configs/experiments/dacc_gate_smoke_r3.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.cli import load_config
from kvcache.provenance import identity, write_gate

MARKER = "dacc-gate-smoke-PASS.json"
FAIL = "dacc-gate-smoke-FAIL.json"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true", help="ignored (launcher compat)")
    p.add_argument("--seeds", default="", help="ignored (launcher compat)")
    return p


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan = {
        "command": "dacc-gate-smoke",
        "model": cfg["models"][0],
        "presses": cfg["presses"],
        "budget": cfg["budget"],
        "onset": cfg["onset"],
        "seed": cfg["seed"],
        "max_new_tokens": cfg["max_new_tokens"],
        "generations": 2 * len(cfg["presses"]) + 2,
        "marker": MARKER,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: loads the model on GPU; pass --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: GPU smoke is remote-only (set KVCACHE_REMOTE=1)")
    base = os.environ.get("KVCACHE_BASE_RUNS_ROOT")
    if not base:
        raise SystemExit("REFUSED: KVCACHE_BASE_RUNS_ROOT is required for the marker")

    from kvcache.datasets import load_benchmark
    from kvcache.generation import (
        exposure_from_events,
        generate_compressed,
        generate_full_cache,
    )
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_cfg = models_cfg[cfg["models"][0]]
    model, tokenizer = load_model_and_tokenizer(model_cfg, allow_download=False)
    problem = load_benchmark(
        "math500", benchmarks_cfg["math500"], subset=1, allow_download=False
    )[0]
    seed, budget, cap = int(cfg["seed"]), int(cfg["budget"]), int(cfg["max_new_tokens"])
    interval = int(cfg["compression_interval"])

    def run(press_name: str, *, gated: bool, start: int | None):
        press = build_press(
            press_name,
            budget=budget,
            compression_interval=interval,
            seed=press_seed(problem.pid, seed),
            instrumented=gated,
        )
        if gated:
            press.compression_start_step = start
        gen = generate_compressed(
            model,
            tokenizer,
            problem,
            model_cfg,
            sampling_cfg,
            press=press,
            seed=seed,
            max_new_tokens=cap,
            step_gate=gated,
        )
        return gen, press

    checks: list[dict[str, Any]] = []
    for press_name in cfg["presses"]:
        legacy, _ = run(press_name, gated=False, start=None)
        gated, press = run(press_name, gated=True, start=0)
        info = exposure_from_events(press.compressions, gated.n_tokens)
        same = legacy.trace_token_ids == gated.trace_token_ids
        checks.append(
            {
                "check": f"parity-{press_name}-{budget}-t0=0",
                "passed": bool(same and info["exposed"]),
                "token_identical": same,
                "n_tokens": [legacy.n_tokens, gated.n_tokens],
                **info,
            }
        )

    onset = int(cfg["onset"])
    full = generate_full_cache(
        model,
        tokenizer,
        problem,
        model_cfg,
        sampling_cfg,
        seed=seed,
        max_new_tokens=cap,
    )
    late, press = run(cfg["presses"][0], gated=True, start=onset)
    info = exposure_from_events(press.compressions, late.n_tokens)
    expected = onset + interval
    ev = info["eviction_onset_step"]
    prefix = ev is not None and late.trace_token_ids[:ev] == full.trace_token_ids[:ev]
    checks.append(
        {
            "check": f"onset-{cfg['presses'][0]}-{budget}-t0={onset}",
            "passed": bool(
                ev == expected
                and prefix
                and info["achieved_exposure"] == late.n_tokens - expected
                and all(e["step"] >= onset for e in press.compressions)
            ),
            "expected_eviction_onset": expected,
            "prefix_identical_to_full_cache": prefix,
            "first_divergence_from_full": next(
                (
                    i
                    for i, (a, b) in enumerate(
                        zip(late.trace_token_ids, full.trace_token_ids, strict=False)
                    )
                    if a != b
                ),
                None,
            ),
            "n_tokens": [full.n_tokens, late.n_tokens],
            **info,
        }
    )

    passed = all(c["passed"] for c in checks)
    report = {
        "identity": identity(
            "dacc", cfg, model_revision=str(model_cfg["revision"]), reference_run=None
        ),
        "gate": "Amendment R.3 D_acc onset gate (real model)",
        "pid": problem.pid,
        "passed": passed,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    target = write_gate(Path(base), MARKER, FAIL, report)
    print(f"{'PASS' if passed else 'FAIL'}: wrote {target}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

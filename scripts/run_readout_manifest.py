#!/usr/bin/env python3
"""Forward readout over a FROZEN manifest split, for the registered Meeting 5
replication (M5-B; prepared 2026-11-10). HELD, and the design is PENDING the
GSM8K timing read.

``scripts/run_replay.py`` is not modified. As ``run_replay_pp.py``,
``run_replay_protected.py`` and ``run_replay_rkv_pp.py`` do, this runner
validates the config, swaps module-level functions on ``run_replay``, and runs
its ``main`` unchanged:

* ``load_benchmark`` -> the split's exact frozen IDs from the pinned pool, so
  the reference rows replayed are exactly the manifest's (``run_replay`` keeps
  only reference rows whose pid it loaded);
* ``build_replay_press`` / ``press_needs_eager_attention`` -> the config's
  family factory, imported from the existing family wrapper (base presses keep
  ``run_replay``'s own);
* ``_load_jsonl`` -> the reference bank's live result files only. A manifest
  bank publishes through the M5 pair path, whose quarantine records are not
  result rows; the bank is first validated as one complete pair per expected
  key (``gsm8k_timing.validate_coverage``) or the readout refuses.

Readout records keep ``run_replay``'s schema; their ``readout_correct`` is the
MATH grader and is never the GSM8K score (``kvcache.manifest_reads``).

    python scripts/run_readout_manifest.py \\
        --config configs/experiments/readout_gsm8k_template_m5b.yaml --dry-run
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_replay

from kvcache import datasets
from kvcache.cli import load_config
from kvcache.gsm8k_timing import validate_coverage
from kvcache.manifest_grid import (
    Family,
    family_of,
    needs_eager,
    require_executable,
    resolve_grid_design,
    run_gate,
)
from kvcache.manifest_problems import select_frozen_split


def replay_factory(family: Family) -> Any:
    """The family's instrumented replay press factory, from its wrapper."""
    if family.name == "pp":
        import run_replay_pp

        return run_replay_pp.build_replay_press_pp
    if family.name == "protsel":
        import run_replay_protected

        return run_replay_protected.build_replay_press_protected
    if family.name == "rkv_pp":
        import run_replay_rkv_pp

        return run_replay_rkv_pp.build_replay_press_rkv_pp
    return run_replay.build_replay_press


def frozen_loader(design: dict[str, Any]) -> Any:
    """``load_benchmark``'s signature, returning exactly the frozen split."""

    def load(bench: str, bench_cfg: Any, **kwargs: Any) -> list[Any]:
        if bench != design["benchmark"]:
            raise SystemExit(f"REFUSED: readout asked for {bench!r}")
        pool = datasets.load_benchmark(
            bench, bench_cfg, subset=None, allow_download=False
        )
        return select_frozen_split(pool, design["pids"], n_pool=design["n_pool"])

    return load


def live_rows_loader(bench: str) -> Any:
    """``_load_jsonl``'s signature, reading only ``<root>/<bench>/**``."""

    def load(root: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for path in sorted((Path(root) / bench).rglob("*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return load


def main() -> int:
    args = run_replay._parser().parse_args()
    config = load_config(args.config)
    try:
        design = resolve_grid_design(config, kind="readout")
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    family = family_of(config)
    base_runs = os.environ.get("KVCACHE_BASE_RUNS_ROOT")
    dry = bool(getattr(args, "dry_run", False))
    run_gate(family, config, dry_run=True, runs_root=base_runs, replay=True)
    if not dry:
        require_executable(config)
        run_gate(family, config, dry_run=False, runs_root=base_runs, replay=True)
        if base_runs is None:
            raise SystemExit("REFUSED: KVCACHE_BASE_RUNS_ROOT is required")
        reference = Path(base_runs) / design["reference_run_name"]
        try:
            validate_coverage(
                reference,
                bench=design["benchmark"],
                pids=design["pids"],
                seeds=design["seeds"],
                manifest_split={
                    "split": design["split"],
                    "revision": design["revision"],
                },
            )
        except ValueError as exc:
            raise SystemExit(f"REFUSED: reference bank: {exc}") from exc
    run_replay.load_benchmark = frozen_loader(design)
    run_replay._load_jsonl = live_rows_loader(design["benchmark"])
    run_replay.build_replay_press = replay_factory(family)
    run_replay.press_needs_eager_attention = lambda p: needs_eager(family, p)
    return run_replay.main()


if __name__ == "__main__":
    raise SystemExit(main())

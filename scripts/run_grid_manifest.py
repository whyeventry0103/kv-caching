#!/usr/bin/env python3
"""Compressed free runs over a FROZEN manifest split, for the registered
Meeting 5 replication (M5-B; prepared 2026-11-10). HELD, and the design is
PENDING the GSM8K timing read: real execution refuses until the config is
unheld, authorized and ``design_status: final``, and its press family's own
gate passes.

Problems are the split's exact frozen IDs (``kvcache.manifest_problems``),
never a count or a sample. Presses come from the config's single registered
family, built by that family's unchanged builder with the per-(problem, seed)
press seed (``kvcache.manifest_grid``). Generation is the package's
``generate_compressed``; ``run_grid.py``, ``run_grid_pp.py`` and
``run_grid_m5.py`` are not modified (their bytes are generation identity for
finished cells). Each condition's row and token-ID sidecar are published as
one recoverable pair through the Meeting 5 path (``kvcache.manifest_publish``);
recovery runs before the model loads and ``--resume`` skips complete pairs.

    python scripts/run_grid_manifest.py \\
        --config configs/experiments/grid_gsm8k_template_m5b.yaml --dry-run
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kvcache.bank import build_result_row
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.generation import generate_compressed
from kvcache.io_schema import validate_row
from kvcache.manifest_grid import (
    builder,
    family_of,
    needs_eager,
    require_executable,
    resolve_grid_design,
    run_gate,
)
from kvcache.manifest_problems import load_frozen_pool, select_frozen_split
from kvcache.manifest_publish import (
    complete_pair_keys,
    pair_identity_problems,
    publish_pair,
    recover_pairs,
)
from kvcache.models import load_model_and_tokenizer
from kvcache.presses import press_seed
from kvcache.provenance import inputs_sha256
from kvcache.runmeta import capture_runmeta
from kvcache.token_ids import build_token_ids_record


def _yaml(path: Path) -> Any:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    return p


def load_problems(design: dict[str, Any]) -> list[Any]:
    """The split's problems from the pinned dataset (offline; no model)."""
    bench_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"][
        design["benchmark"]
    ]
    pool = load_frozen_pool(design["benchmark"], bench_cfg, hf_loader=load_benchmark)
    return select_frozen_split(pool, design["pids"], n_pool=design["n_pool"])


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "grid_manifest")
    try:
        design = resolve_grid_design(config, kind="grid")
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    family = family_of(config)
    base_runs = os.environ.get("KVCACHE_BASE_RUNS_ROOT")
    run_gate(family, config, dry_run=True, runs_root=base_runs, replay=False)
    summary = {k: v for k, v in design.items() if k != "pids"}
    summary.update(n_problems=len(design["pids"]), held=bool(config.get("held", True)))
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    require_executable(config)
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    run_gate(family, config, dry_run=False, runs_root=base_runs, replay=False)

    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    bench, model_name = design["benchmark"], design["model"]
    split_tag = {"split": design["split"], "revision": design["revision"]}
    seeds = design["seeds"]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in shard]
        if not seeds:
            raise SystemExit(f"--seeds {args.seeds} disjoint from config seeds")
    stats = recover_pairs(
        runs_root,
        bench,
        seeds=seeds,
        validate=lambda rec: pair_identity_problems(
            rec,
            bench=bench,
            model=model_name,
            pids=design["pids"],
            seeds=seeds,
            manifest_split=split_tag,
            presses=design["presses"],
            budgets=design["budgets"],
        ),
    )
    print(f"RECOVERY: {json.dumps(stats, sort_keys=True)}", flush=True)

    problems = load_problems(design)
    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_cfg = models_cfg[model_name]
    eager = any(needs_eager(family, p) for p in design["presses"])
    model, tokenizer = load_model_and_tokenizer(
        model_cfg,
        allow_download=False,
        attn_implementation="eager" if eager else None,
    )
    done = complete_pair_keys(runs_root, bench, seeds) if args.resume else set()
    counts = {"rows": 0, "truncated": 0, "skipped_resume": len(done)}
    build = builder(family)
    interval = int(config["compression_interval"])
    cap = design["max_new_tokens"]
    for press_name in design["presses"]:
        for budget in design["budgets"]:
            for seed in seeds:
                runmeta = capture_runmeta(config["name"], args.config, int(seed))
                for problem in problems:
                    key = (press_name, str(budget), problem.pid, int(seed))
                    if key in done:
                        continue
                    press = build(
                        press_name,
                        budget=int(budget),
                        compression_interval=interval,
                        seed=press_seed(problem.pid, int(seed)),
                    )
                    generation = generate_compressed(
                        model,
                        tokenizer,
                        problem,
                        model_cfg,
                        sampling_cfg,
                        press=press,
                        seed=int(seed),
                        max_new_tokens=cap,
                    )
                    n_prompt = len(generation.prompt_token_ids)
                    row = build_result_row(
                        bench=bench,
                        pid=problem.pid,
                        model_name=model_name,
                        seed=int(seed),
                        max_new_tokens=cap,
                        sampling_cfg=sampling_cfg,
                        generation=generation,
                        runmeta=runmeta,
                        press=press_name,
                        budget=int(budget),
                        compression_ratio=min(
                            1.0, int(budget) / (n_prompt + generation.n_tokens)
                        ),
                    )
                    row["manifest_split"] = dict(split_tag)
                    row["press_family"] = family.name
                    row["inputs_sha256"] = inputs_sha256(generation.prompt_token_ids)
                    protect = getattr(press, "protect", None)
                    if protect is not None and hasattr(protect, "report"):
                        row["protection"] = protect.report()
                    validate_row(row)
                    side = build_token_ids_record(
                        result_row=row,
                        prompt_token_ids=generation.prompt_token_ids,
                        trace_token_ids=generation.trace_token_ids,
                        prompt_text=generation.prompt,
                        trace_text=generation.trace,
                        tokenizer=tokenizer,
                        tokenizer_revision=str(model_cfg["revision"]),
                    )
                    side["manifest_split"] = dict(split_tag)
                    output = publish_pair(runs_root, bench, row, side)
                    counts["rows"] += 1
                    counts["truncated"] += int(row["truncated"])
                    print(
                        f"wrote {output}: {bench}/{problem.pid}/s{seed} "
                        f"{press_name}@{budget}",
                        flush=True,
                    )
    if counts["rows"]:
        counts["truncation_rate"] = counts["truncated"] / counts["rows"]
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Full-cache generation over a FROZEN manifest split (Meeting 5 pid-filter
adapter, 2026-11-10; first use: the GSM8K timing proposal). HELD: nothing runs
until the config says ``held: false`` and ``scale_authorized: true``.

Problems come from ``kvcache.manifest_problems``: the split's exact frozen IDs,
checked against the pinned dataset revision and the frozen pool size, in
manifest order. Generation is ``build_bank.py``'s full-cache path, imported
rather than edited (``run_grid.py``/``build_bank.py``/``datasets.py`` stay
byte-identical, so no existing generation identity or generation-path check
moves).

Publication reuses the Meeting 5 runner's recoverable pair path
(``kvcache.manifest_publish`` over the unchanged ``kvcache.m5_publish``): each
condition's row and token-ID sidecar are validated, journalled together and
published idempotently. Before the model loads, pending envelopes are finished
and orphans quarantined in the owned seed shards; ``--resume`` skips complete
pairs only. A stop between the row and sidecar writes therefore never needs a
regeneration and can never leave the run unable to complete.

Modes, all refusing on any mismatch with the frozen design:

* ``--dry-run``: validates the config and resolves the split from the
  manifest alone (no dataset, no model) and checks the declared generation
  count.
* ``--resolve-only``: loads the pinned dataset OFFLINE (no model, no
  download), selects the split and prints each problem's ID and question hash.
  CPU only; the boundary check that the IDs exist in the real pool.
* default: the guarded generation run (``--i-understand``, ``KVCACHE_REMOTE=1``).

    python scripts/run_bank_manifest.py \\
        --config configs/experiments/bank_gsm8k_timing_m5.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
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
from kvcache.generation import generate_full_cache
from kvcache.io_schema import validate_row
from kvcache.manifest_problems import (
    frozen_split_pids,
    load_frozen_pool,
    manifest_key,
    select_frozen_split,
)
from kvcache.manifest_publish import (
    complete_pair_keys,
    pair_identity_problems,
    publish_pair,
    recover_pairs,
)
from kvcache.models import load_model_and_tokenizer
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
    p.add_argument("--resolve-only", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    return p


def resolve_design(config: dict[str, Any]) -> dict[str, Any]:
    """The frozen split and run shape a config declares, or ValueError."""
    if config.get("presses") != ["fullcache"] or config.get("budgets") != ["full"]:
        raise ValueError("this runner generates full-cache banks only")
    if len(config["benchmarks"]) != 1:
        raise ValueError("one benchmark per manifest bank")
    if config.get("store_token_ids") is not True:
        raise ValueError("manifest banks publish row + token-ID pairs")
    bench = config["benchmarks"][0]
    spec = config.get("manifest") or {}
    if not spec.get("path") or not spec.get("split"):
        raise ValueError("config.manifest needs path and split")
    manifest = json.loads((PROJECT_ROOT / spec["path"]).read_text(encoding="utf-8"))
    bench_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"][bench]
    mkey = manifest_key(bench, bench_cfg)
    pids = frozen_split_pids(
        manifest, mkey, spec["split"], pinned_revision=str(bench_cfg["revision"])
    )
    seeds = [int(s) for s in config["seeds"]]
    n_gen = len(pids) * len(seeds)
    if int(config.get("estimated_generation_count", -1)) != n_gen:
        raise ValueError(
            f"estimated_generation_count {config.get('estimated_generation_count')}"
            f" != {len(pids)} problems x {len(seeds)} seeds = {n_gen}"
        )
    return {
        "benchmark": bench,
        "split": spec["split"],
        "pids": pids,
        "n_pool": int(manifest[mkey]["n_pool"]),
        "seeds": seeds,
        "n_generations": n_gen,
        "max_new_tokens": int(config["max_new_tokens"][bench]),
        "revision": str(bench_cfg["revision"]),
        "model": str(config["models"][0]),
    }


def load_split(design: dict[str, Any]) -> list[Any]:
    """The split's problems from the pinned dataset (offline; no model)."""
    bench_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"][
        design["benchmark"]
    ]
    pool = load_frozen_pool(design["benchmark"], bench_cfg, hf_loader=load_benchmark)
    return select_frozen_split(pool, design["pids"], n_pool=design["n_pool"])


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "bank")
    try:
        design = resolve_design(config)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
    summary = {k: v for k, v in design.items() if k != "pids"}
    summary.update(
        n_problems=len(design["pids"]),
        held=bool(config.get("held", True)),
        first_pids=design["pids"][:3],
    )
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if args.resolve_only:
        try:
            problems = load_split(design)
        except ValueError as exc:
            raise SystemExit(f"REFUSED: {exc}") from exc
        print(
            json.dumps(
                {
                    "resolved": len(problems),
                    "problems": [
                        {
                            "pid": p.pid,
                            "question_sha256": hashlib.sha256(
                                p.question.encode()
                            ).hexdigest(),
                            "gold": p.gold_answer,
                        }
                        for p in problems
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if config.get("held", True):
        raise SystemExit("REFUSED: this bank is HELD (needs its own approval)")
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not config.get("scale_authorized"):
        raise SystemExit("REFUSED: config must set scale_authorized: true")

    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    bench, cap, model_name = (
        design["benchmark"],
        design["max_new_tokens"],
        design["model"],
    )
    split_tag = {"split": design["split"], "revision": design["revision"]}
    seeds = design["seeds"]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in shard]
        if not seeds:
            raise SystemExit(f"--seeds {args.seeds} disjoint from config seeds")
    # Finish interrupted pair publications and quarantine orphans in the owned
    # seed shards BEFORE the model loads; "done" then means a complete pair.
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
        ),
    )
    print(f"RECOVERY: {json.dumps(stats, sort_keys=True)}", flush=True)

    problems = load_split(design)
    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_cfg = models_cfg[model_name]
    model, tokenizer = load_model_and_tokenizer(model_cfg, allow_download=False)
    done = complete_pair_keys(runs_root, bench, seeds) if args.resume else set()
    counts = {"rows": 0, "truncated": 0, "skipped_resume": len(done)}
    for seed in seeds:
        runmeta = capture_runmeta(config["name"], args.config, int(seed))
        for problem in problems:
            if ("fullcache", "full", problem.pid, int(seed)) in done:
                continue
            generation = generate_full_cache(
                model,
                tokenizer,
                problem,
                model_cfg,
                sampling_cfg,
                seed=int(seed),
                max_new_tokens=cap,
            )
            row = build_result_row(
                bench=bench,
                pid=problem.pid,
                model_name=model_name,
                seed=int(seed),
                max_new_tokens=cap,
                sampling_cfg=sampling_cfg,
                generation=generation,
                runmeta=runmeta,
            )
            row["manifest_split"] = dict(split_tag)
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
            # both records validated -> one atomic envelope -> publish
            output = publish_pair(runs_root, bench, row, side)
            counts["rows"] += 1
            counts["truncated"] += int(row["truncated"])
            print(f"wrote {output}: {bench}/{problem.pid}/s{seed}", flush=True)
    counts["truncation_rate"] = (
        counts["truncated"] / counts["rows"] if counts["rows"] else 0.0
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

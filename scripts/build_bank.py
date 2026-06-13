#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.bank import append_result, build_result_row
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.generation import generate_full_cache
from kvcache.models import (
    load_model_and_tokenizer,
    load_tokenizer,
    render_prompt,
)
from kvcache.runmeta import capture_runmeta


def _yaml(path: Path):
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full-cache reference banks")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-template", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--allow-dataset-download", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "bank")
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not config.get("pilot"):
        raise SystemExit("REFUSED: only pilot_fullcache.yaml is executable")

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_name = config["models"][0]
    model_cfg = models_cfg[model_name]

    if args.probe_template:
        print("Action: load tokenizer and render one user-only chat template")
        tokenizer = load_tokenizer(
            model_cfg,
            allow_download=args.allow_model_download,
        )
        print(render_prompt(tokenizer, "What is 1 + 1?", model_cfg))
        return 0

    print("Action: load one model and generate the guarded full-cache pilot")
    print(f"Model download allowed: {args.allow_model_download}")
    print(f"Dataset download allowed: {args.allow_dataset_download}")
    model, tokenizer = load_model_and_tokenizer(
        model_cfg,
        allow_download=args.allow_model_download,
    )
    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    counts = {"rows": 0, "truncated": 0}

    for benchmark in config["benchmarks"]:
        subset = config.get("subset", {}).get(benchmark)
        problems = load_benchmark(
            benchmark,
            benchmarks_cfg[benchmark],
            subset=subset,
            allow_download=args.allow_dataset_download,
        )
        configured_max_new = config["max_new_tokens"]
        max_new_tokens = (
            int(configured_max_new[benchmark])
            if isinstance(configured_max_new, dict)
            else int(configured_max_new)
        )
        for seed in config["seeds"]:
            runmeta = capture_runmeta(config["name"], args.config, int(seed))
            for problem in problems:
                generation = generate_full_cache(
                    model,
                    tokenizer,
                    problem,
                    model_cfg,
                    sampling_cfg,
                    seed=int(seed),
                    max_new_tokens=max_new_tokens,
                )
                row = build_result_row(
                    bench=benchmark,
                    pid=problem.pid,
                    model_name=model_name,
                    seed=int(seed),
                    max_new_tokens=max_new_tokens,
                    sampling_cfg=sampling_cfg,
                    generation=generation,
                    runmeta=runmeta,
                )
                output = append_result(runs_root, row)
                counts["rows"] += 1
                counts["truncated"] += int(row["truncated"])
                print(f"wrote {output}: {benchmark}/{problem.pid}/s{seed}")
    counts["truncation_rate"] = (
        counts["truncated"] / counts["rows"] if counts["rows"] else 0.0
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

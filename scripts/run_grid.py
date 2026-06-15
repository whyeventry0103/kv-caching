#!/usr/bin/env python3
"""Run the guarded compressed-generation pilot grid."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.bank import append_result, build_result_row
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.generation import generate_compressed
from kvcache.models import (
    load_model_and_tokenizer,
    render_prompt,
    tokenize_rendered_prompt,
)
from kvcache.presses import build_press, press_needs_eager_attention, press_seed
from kvcache.runmeta import capture_runmeta


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "grid")
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not config.get("pilot") or config.get("estimated_generation_count") != 80:
        raise SystemExit(
            "REFUSED: only the reviewed 80-row compression pilot is eligible"
        )

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_name = config["models"][0]
    model_cfg = models_cfg[model_name]
    eager = any(press_needs_eager_attention(p) for p in config["presses"])
    model, tokenizer = load_model_and_tokenizer(
        model_cfg, attn_implementation="eager" if eager else None
    )
    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    counts = {"rows": 0, "truncated": 0}

    for benchmark in config["benchmarks"]:
        problems = load_benchmark(
            benchmark,
            benchmarks_cfg[benchmark],
            subset=config["subset"][benchmark],
            allow_download=False,
        )
        max_new_tokens = int(config["max_new_tokens"][benchmark])
        for press_name in config["presses"]:
            for budget in config["budgets"]:
                for seed in config["seeds"]:
                    runmeta = capture_runmeta(config["name"], args.config, int(seed))
                    for problem in problems:
                        press = build_press(
                            press_name,
                            budget=int(budget),
                            compression_interval=int(config["compression_interval"]),
                            # Per-(problem, seed) RNG so random eviction is
                            # decorrelated across problems (no lucky-seed artifact).
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
                            max_new_tokens=max_new_tokens,
                        )
                        prompt_tokens = len(
                            tokenize_rendered_prompt(
                                tokenizer,
                                render_prompt(tokenizer, problem.question, model_cfg),
                            )["input_ids"]
                        )
                        logical_tokens = prompt_tokens + generation.n_tokens
                        row = build_result_row(
                            bench=benchmark,
                            pid=problem.pid,
                            model_name=model_name,
                            seed=int(seed),
                            max_new_tokens=max_new_tokens,
                            sampling_cfg=sampling_cfg,
                            generation=generation,
                            runmeta=runmeta,
                            press=press_name,
                            budget=int(budget),
                            compression_ratio=min(1.0, int(budget) / logical_tokens),
                        )
                        output = append_result(runs_root, row)
                        counts["rows"] += 1
                        counts["truncated"] += int(row["truncated"])
                        print(f"wrote {output}: {benchmark}/{problem.pid}/s{seed}")
    counts["truncation_rate"] = counts["truncated"] / counts["rows"]
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

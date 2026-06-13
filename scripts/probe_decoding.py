#!/usr/bin/env python3
"""One-row DecodingPress diagnostic with measured cache lengths."""

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
from kvcache.generation import generate_with_press
from kvcache.models import load_model_and_tokenizer
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
    plan(config, args.config, "probe-decoding")
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if config.get("estimated_generation_count") != 1:
        raise SystemExit("REFUSED: decoding probe must contain exactly one generation")

    from kvpress import DecodingPress, StreamingLLMPress

    class InstrumentedDecodingPress(DecodingPress):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.compressions: list[dict[str, int]] = []

        def compress(self, module, hidden_states, keys, values, attentions, kwargs):
            before = int(keys.shape[2])
            compressed_keys, compressed_values = super().compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
            self.compressions.append(
                {
                    "layer": int(module.layer_idx),
                    "before": before,
                    "after": int(compressed_keys.shape[2]),
                }
            )
            return compressed_keys, compressed_values

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_name = config["models"][0]
    benchmark = config["benchmarks"][0]
    seed = int(config["seeds"][0])
    budget = int(config["budgets"][0])
    interval = int(config["compression_interval"])
    max_new_tokens = int(config["max_new_tokens"])
    problem = load_benchmark(
        benchmark,
        benchmarks_cfg[benchmark],
        subset=config["subset"][benchmark],
        allow_download=False,
    )[0]
    model, tokenizer = load_model_and_tokenizer(models_cfg[model_name])
    press = InstrumentedDecodingPress(
        base_press=StreamingLLMPress(),
        target_size=budget,
        compression_interval=interval,
        hidden_states_buffer_size=0,
    )
    generation, prompt_tokens = generate_with_press(
        model,
        tokenizer,
        problem,
        models_cfg[model_name],
        sampling_cfg,
        press=press,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    logical_tokens = prompt_tokens + generation.n_tokens
    compression_ratio = min(1.0, budget / logical_tokens)
    runmeta = capture_runmeta(config["name"], args.config, seed)
    row = build_result_row(
        bench=benchmark,
        pid=problem.pid,
        model_name=model_name,
        seed=seed,
        max_new_tokens=max_new_tokens,
        sampling_cfg=sampling_cfg,
        generation=generation,
        runmeta=runmeta,
        press="streamingllm",
        budget=budget,
        compression_ratio=compression_ratio,
    )
    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    output = append_result(runs_root, row)
    stats = {
        "budget_semantics": "exact retained KV tokens immediately after each prune",
        "target_size": budget,
        "compression_interval": interval,
        "expected_between_prune_max": budget + interval,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generation.n_tokens,
        "logical_tokens": logical_tokens,
        "compression_events": len(press.compressions),
        "max_before_prune": max(
            (item["before"] for item in press.compressions), default=logical_tokens
        ),
        "max_after_prune": max(
            (item["after"] for item in press.compressions), default=logical_tokens
        ),
        "all_after_at_most_target": all(
            item["after"] <= budget for item in press.compressions
        ),
        "events": press.compressions,
    }
    stats_path = runs_root / "cache_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

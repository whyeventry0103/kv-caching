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
from kvcache.generation import (
    STEP_GATE_VERSION,
    exposure_from_events,
    generate_compressed,
)
from kvcache.models import (
    load_model_and_tokenizer,
    render_prompt,
    tokenize_rendered_prompt,
)
from kvcache.presses import build_press, press_needs_eager_attention, press_seed
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    same_identity,
)
from kvcache.runmeta import capture_runmeta
from kvcache.token_ids import append_token_ids, build_token_ids_record


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip (bench,pid,press,budget,seed) already generated in the output dir",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated subset of config seeds to generate (per-GPU sharding: "
        "each shard owns disjoint per-seed output files, so two GPUs never collide)",
    )
    return parser


def _grid_done_keys(runs_root: Path, store_token_ids: bool) -> set:
    """Completed (bench, pid, press, budget, seed) keys; heal partial trailing lines.

    With store_token_ids, a key is done only if it has BOTH a result row and a token-ID
    sidecar -- a row whose sidecar is missing is incomplete and gets redone on resume.
    """

    def scan(paths: list[Path]) -> set:
        keys = set()
        for path in paths:
            lines = path.read_text(encoding="utf-8").splitlines()
            valid = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                valid.append(line)
                seed = row.get("decoding", {}).get("seed", row.get("seed"))
                keys.add(
                    (
                        row["bench"],
                        row["pid"],
                        row["press"],
                        str(row["budget"]),
                        int(seed),
                    )
                )
            if len(valid) != len([x for x in lines if x.strip()]):
                healed = ("\n".join(valid) + "\n") if valid else ""
                path.write_text(healed, encoding="utf-8")
        return keys

    all_jsonl = sorted(runs_root.rglob("*.jsonl"))
    result_keys = scan([p for p in all_jsonl if "token_ids" not in p.parts])
    if not store_token_ids:
        return result_keys
    return result_keys & scan([p for p in all_jsonl if "token_ids" in p.parts])


def _onset_gate(config: dict[str, Any]) -> tuple[bool, int | None]:
    """Amendment R.3: whether this grid is onset-gated, and its start step.

    The KEY's presence selects the gated path (instrumented press driven by a
    decode-step pre-hook); a value of null is the always-on gate that the
    parity test pins to legacy generation. Absent key = legacy path, untouched.
    """
    if "compression_start_step" not in config:
        return False, None
    start = config["compression_start_step"]
    if start is not None and (
        isinstance(start, bool) or not isinstance(start, int) or start < 0
    ):
        raise SystemExit(
            f"REFUSED: compression_start_step must be a non-negative int or null, "
            f"got {start!r}"
        )
    return True, start


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "grid")
    gated, start_step = _onset_gate(config)
    if gated:
        print(
            json.dumps(
                {"onset_gate": STEP_GATE_VERSION, "compression_start_step": start_step}
            )
        )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not (config.get("pilot") or config.get("scale_authorized")):
        raise SystemExit(
            "REFUSED: grid config must set pilot:true or scale_authorized:true"
        )
    if config.get("pilot") and config.get("estimated_generation_count") != 80:
        raise SystemExit("REFUSED: the pilot grid must be exactly 80 rows")

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
    store_token_ids = bool(config.get("store_token_ids"))
    config_seeds = [int(s) for s in config["seeds"]]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        active_seeds = [s for s in config_seeds if s in shard]
        if not active_seeds:
            raise SystemExit(
                f"--seeds {args.seeds} disjoint from config seeds {config_seeds}"
            )
    else:
        active_seeds = config_seeds
    ident = (
        identity("dacc", config, model_revision=str(model_cfg["revision"]))
        if gated
        else None
    )
    manifest = load_input_manifest() if gated else None
    if gated:
        # Onset-gated (D_acc) rows carry an experiment identity; a namespace
        # holding rows of another identity is refused, never resumed into.
        for path in sorted(runs_root.rglob("*.jsonl")):
            if "token_ids" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not same_identity(rec.get("provenance") or {}, ident):
                    raise SystemExit(
                        f"REFUSED: {path} holds a row from a different experiment "
                        "identity; start a new run namespace"
                    )
                check_inputs(
                    manifest,
                    "dacc",
                    rec["pid"],
                    int(rec["decoding"]["seed"]),
                    rec["provenance"].get("inputs_sha256"),
                )
    done_keys = _grid_done_keys(runs_root, store_token_ids) if args.resume else set()
    counts = {"rows": 0, "truncated": 0, "skipped_resume": len(done_keys)}
    if args.resume:
        print(f"RESUME: skipping {len(done_keys)} already-complete generations")

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
                for seed in active_seeds:
                    runmeta = capture_runmeta(config["name"], args.config, int(seed))
                    for problem in problems:
                        if (
                            benchmark,
                            problem.pid,
                            press_name,
                            str(budget),
                            int(seed),
                        ) in done_keys:
                            continue
                        press = build_press(
                            press_name,
                            budget=int(budget),
                            compression_interval=int(config["compression_interval"]),
                            # Per-(problem, seed) RNG so random eviction is
                            # decorrelated across problems (no lucky-seed artifact).
                            seed=press_seed(problem.pid, int(seed)),
                            instrumented=gated,
                        )
                        if gated:
                            press.compression_start_step = start_step
                        generation = generate_compressed(
                            model,
                            tokenizer,
                            problem,
                            model_cfg,
                            sampling_cfg,
                            press=press,
                            seed=int(seed),
                            max_new_tokens=max_new_tokens,
                            step_gate=gated,
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
                        if gated:
                            row["onset"] = {
                                "gate": STEP_GATE_VERSION,
                                "compression_start_step": start_step,
                                "prune_events": len(press.compressions),
                                **exposure_from_events(
                                    press.compressions, generation.n_tokens
                                ),
                            }
                            got = inputs_sha256(generation.prompt_token_ids)
                            check_inputs(manifest, "dacc", problem.pid, int(seed), got)
                            row["provenance"] = {**ident, "inputs_sha256": got}
                        output = append_result(runs_root, row)
                        if store_token_ids:
                            # Exact token IDs make every compressed trace replay-ready
                            # (reverse readout, TV profiles) without lossy re-tokenize.
                            append_token_ids(
                                runs_root,
                                build_token_ids_record(
                                    result_row=row,
                                    prompt_token_ids=generation.prompt_token_ids,
                                    trace_token_ids=generation.trace_token_ids,
                                    prompt_text=generation.prompt,
                                    trace_text=generation.trace,
                                    tokenizer=tokenizer,
                                    tokenizer_revision=str(model_cfg["revision"]),
                                ),
                            )
                        counts["rows"] += 1
                        counts["truncated"] += int(row["truncated"])
                        print(f"wrote {output}: {benchmark}/{problem.pid}/s{seed}")
    counts["truncation_rate"] = counts["truncated"] / counts["rows"]
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

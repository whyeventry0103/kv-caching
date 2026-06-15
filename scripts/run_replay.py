#!/usr/bin/env python3
"""Run guarded pilot teacher-forced replay or readout evaluation."""

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

from kvcache.bank import append_result
from kvcache.canonicalize import extract_answer
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.determinism import log_versions
from kvcache.models import load_model_and_tokenizer
from kvcache.presses import build_replay_press, press_seed
from kvcache.replay import (
    TransformersReplayBackend,
    build_replay_row,
    paired_teacher_forced_replay,
    reasoning_boundary_token,
    score_readout_answer,
    teacher_forced_readout_tokens,
    write_replay_profile,
)
from kvcache.runmeta import capture_runmeta
from kvcache.token_ids import validate_token_ids_record


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


def _load_jsonl(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for path in sorted(root.rglob("*.jsonl"))
        if "token_ids" not in path.parts
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _load_token_records(root: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    records = {}
    for path in sorted((root / "token_ids").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            key = (
                record["bench"],
                record["pid"],
                record["model"],
                record["press"],
                record["budget"],
                int(record["seed"]),
            )
            if key in records:
                raise SystemExit(f"duplicate token-ID record: {key}")
            records[key] = record
    return records


def _reference_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["bench"],
        row["pid"],
        row["model"],
        row["press"],
        row["budget"],
        int(row["decoding"]["seed"]),
    )


def _readout_path(root: Path, press: str, budget: int, seed: int) -> Path:
    return root / "readout" / f"{press}-{budget}-s{seed}.jsonl"


def _eos_token_ids(tokenizer: Any) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        raise SystemExit("tokenizer must define eos_token_id for readout")
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "replay")
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not config.get("pilot") or config.get("mode") not in {
        "self_replay",
        "compression_replay",
        "readout",
    }:
        raise SystemExit("REFUSED: only reviewed M4 pilot modes are executable")

    expected = int(config["estimated_replay_count"])
    if expected not in {1, 20, 80}:
        raise SystemExit("REFUSED: M4 pilot must contain 1, 20, or 80 conditions")
    base_runs = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    reference_root = base_runs / config["reference_run_name"]
    if not reference_root.is_dir():
        raise SystemExit(f"missing reference run: {reference_root}")
    reference_rows = _load_jsonl(reference_root)
    token_records = _load_token_records(reference_root)
    if len(reference_rows) != 20 or len(token_records) != 20:
        raise SystemExit("M4 reference must contain exactly 20 rows and token records")

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_name = config["models"][0]
    model_cfg = models_cfg[model_name]
    installed = log_versions()
    for row in reference_rows:
        for package in ("torch", "transformers", "kvpress"):
            if row["versions"][package] != installed[package]:
                raise SystemExit(f"reference {package} version does not match runtime")
    problems = load_benchmark(
        "math500",
        benchmarks_cfg["math500"],
        subset={"count": 5, "stratify_by": "level"},
        allow_download=False,
    )
    gold_by_pid = {problem.pid: problem.gold_answer for problem in problems}
    model, tokenizer = load_model_and_tokenizer(model_cfg, allow_download=False)
    output_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    mode = config["mode"]
    completed = 0

    eligible_rows = [
        row
        for row in reference_rows
        if row["model"] == model_name and row["decoding"]["seed"] in config["seeds"]
    ]
    reference_limit = config.get("reference_limit")
    if reference_limit is not None:
        eligible_rows = eligible_rows[: int(reference_limit)]

    for row in eligible_rows:
        token_record = token_records.get(_reference_key(row))
        if token_record is None:
            raise SystemExit(f"missing token IDs for {_reference_key(row)}")
        decoded_prompt = tokenizer.decode(
            token_record["prompt_token_ids"], skip_special_tokens=False
        )
        validate_token_ids_record(
            token_record,
            result_row=row,
            prompt_text=decoded_prompt,
            trace_text=row["trace"],
            tokenizer=tokenizer,
        )
        if token_record["tokenizer_revision"] != str(model_cfg["revision"]):
            raise SystemExit("reference tokenizer revision does not match runtime")
        prompt_ids = token_record["prompt_token_ids"]
        trace_ids = token_record["trace_token_ids"]
        seed = int(row["decoding"]["seed"])
        for press_name in config["presses"]:
            for budget in config["budgets"]:
                press = build_replay_press(
                    press_name,
                    budget=2048 if budget == "full" else int(budget),
                    compression_interval=int(config["compression_interval"]),
                    # Per-(problem, seed) RNG (decorrelate random eviction).
                    seed=press_seed(row["pid"], seed),
                )
                backend = TransformersReplayBackend(model)
                runmeta = capture_runmeta(config["name"], args.config, seed)
                if mode == "readout":
                    import torch

                    boundary = reasoning_boundary_token(
                        tokenizer, trace_ids, row["trace"]
                    )
                    with torch.inference_mode(), press(model):
                        generated_ids = teacher_forced_readout_tokens(
                            prompt_ids,
                            trace_ids,
                            boundary=boundary,
                            cache_factory=backend.cache_factory,
                            forward=backend.forward,
                            press=press,
                            seed=seed,
                            temperature=float(sampling_cfg["temperature"]),
                            top_p=float(sampling_cfg["top_p"]),
                            eos_token_ids=_eos_token_ids(tokenizer),
                            max_answer_tokens=int(config["max_answer_tokens"]),
                        )
                    generated_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    answer_raw = extract_answer(generated_text)
                    record = {
                        "schema_version": "readout-v1",
                        "reference_key": list(_reference_key(row)),
                        "condition": {"press": press_name, "budget": budget},
                        "boundary_token": boundary,
                        "generated_token_ids": generated_ids,
                        "generated_text": generated_text,
                        "answer_raw": answer_raw,
                        "readout_correct": score_readout_answer(
                            gold_by_pid[row["pid"]], generated_text
                        ),
                        "runmeta": runmeta,
                        "versions": row["versions"],
                    }
                    path = _readout_path(output_root, press_name, int(budget), seed)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        for line in path.read_text(encoding="utf-8").splitlines():
                            existing = json.loads(line)
                            if existing["reference_key"] == record["reference_key"]:
                                raise FileExistsError(
                                    f"readout result already exists: {path}"
                                )
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                else:
                    with __import__("torch").inference_mode(), press(model):
                        profile = paired_teacher_forced_replay(
                            prompt_ids,
                            trace_ids,
                            cache_factory=backend.cache_factory,
                            forward=backend.forward,
                            press=press,
                        )
                    if mode == "self_replay" and max(profile.estimate) > 1e-6:
                        raise SystemExit("self-replay TV exceeded 1e-6")
                    if profile.compression_events and budget != "full":
                        if profile.max_after_prune is None or (
                            profile.max_after_prune > int(budget)
                        ):
                            raise SystemExit("post-prune cache exceeded target budget")
                        if profile.max_before_prune is None or (
                            profile.max_before_prune
                            > int(budget) + int(config["compression_interval"])
                        ):
                            raise SystemExit(
                                "between-prune cache exceeded expected bound"
                            )
                    if profile.onset_step is not None and any(
                        value > 1e-6 for value in profile.estimate[: profile.onset_step]
                    ):
                        raise SystemExit(
                            "compressed replay diverged before first prune"
                        )
                    pid_hash = hashlib.sha256(row["pid"].encode()).hexdigest()[:16]
                    profile_path = write_replay_profile(
                        output_root,
                        reference_key=_reference_key(row),
                        press_name=press_name,
                        budget=budget,
                        pid_hash=pid_hash,
                        profile=profile,
                    )
                    logical_tokens = len(prompt_ids) + len(trace_ids)
                    ratio = (
                        1.0
                        if budget == "full"
                        else min(1.0, int(budget) / logical_tokens)
                    )
                    replay_row = build_replay_row(
                        row,
                        press_name=press_name,
                        budget=budget,
                        compression_ratio=ratio,
                        replay_of=f"{config['reference_run_name']}:{_reference_key(row)}",
                        tv_profile_path=str(profile_path.relative_to(output_root)),
                        profile=profile,
                        runmeta=runmeta,
                    )
                    append_result(output_root, replay_row)
                completed += 1

    if completed != expected:
        raise SystemExit(
            f"expected {expected} replay conditions, completed {completed}"
        )
    print(json.dumps({"mode": mode, "completed": completed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

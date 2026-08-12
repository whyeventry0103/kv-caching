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
from kvcache.presses import (
    build_ratio_matched_press,
    build_replay_press,
    build_restoration_press,
    press_needs_eager_attention,
    press_seed,
)
from kvcache.metrics.restoration import REINSERT_FRACTION, TRIGGER_STEP
from kvcache.metrics.divergence import measures_from_logits
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


def _condition_arms(config: dict[str, Any]) -> list[str | None]:
    """The restoration arms this config runs, or ``[None]`` for a plain replay.

    ``none`` is required whenever any arm runs: the effect of an intervention is
    only defined against the no-intervention arm on the same traces, so a config
    that omits it cannot produce a readable result (Amendment P.5).
    """
    restoration = config.get("restoration")
    if not restoration:
        return [None]
    arms = list(restoration.get("arms") or [])
    if not arms:
        raise SystemExit("restoration config lists no arms")
    if "none" not in arms:
        raise SystemExit(
            "the 'none' arm is required: without the temporal null a post-trigger "
            "decline cannot be distinguished from what would have happened anyway"
        )
    if config.get("matched_ratio") is not None:
        raise SystemExit(
            "matched_ratio and restoration cannot run in one config: the "
            "intervention and the budget schedule would be confounded"
        )
    return arms


def _build_condition_press(
    press_name: str,
    *,
    budget: int | str,
    config: dict[str, Any],
    seed: int,
    arm: str | None,
) -> Any:
    """Select the press for this condition: plain replay, M.3 ratio-matched, or P.5."""
    interval = int(config["compression_interval"])
    effective_budget = 2048 if budget == "full" else int(budget)
    if arm is not None:
        restoration = config["restoration"]
        return build_restoration_press(
            press_name,
            budget=effective_budget,
            arm=arm,
            trigger_step=int(restoration.get("trigger_step", TRIGGER_STEP)),
            reinsert_fraction=float(
                restoration.get("reinsert_fraction", REINSERT_FRACTION)
            ),
            compression_interval=interval,
            seed=seed,
        )
    matched_ratio = config.get("matched_ratio")
    if matched_ratio is not None:
        return build_ratio_matched_press(
            press_name,
            ratio=float(matched_ratio),
            min_budget=int(config.get("matched_min_budget", 32)),
            compression_interval=interval,
            seed=seed,
        )
    return build_replay_press(
        press_name,
        budget=effective_budget,
        compression_interval=interval,
        seed=seed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-understand", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip (reference_key, press, budget) conditions already written under "
        "the output dir; heal any partial trailing line left by a timed stop",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated subset of config seeds to process (per-GPU sharding: "
        "each shard owns disjoint per-seed output files, so two GPUs never collide)",
    )
    return parser


def _heal_jsonl(path: Path) -> list[dict[str, Any]]:
    """Return parsed records, rewriting the file to drop a partial trailing line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    raw = [line for line in lines if line.strip()]
    valid = []
    parsed = []
    for line in raw:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        valid.append(line)
    if len(valid) != len(raw):
        path.write_text(("\n".join(valid) + "\n") if valid else "", encoding="utf-8")
    return parsed


def _replay_done_keys(output_root: Path, mode: str) -> set:
    """Completed (reference_key, press, budget) condition keys under the output dir."""
    done = set()
    if mode == "readout":
        readout_dir = output_root / "readout"
        if not readout_dir.exists():
            return done
        for path in sorted(readout_dir.rglob("*.jsonl")):
            for rec in _heal_jsonl(path):
                done.add(
                    (
                        tuple(rec["reference_key"]),
                        rec["condition"]["press"],
                        str(rec["condition"]["budget"]),
                    )
                )
        return done
    for path in sorted(output_root.rglob("*.jsonl")):
        parts = path.relative_to(output_root).parts
        if parts and parts[0] in {"readout", "profiles", "token_ids"}:
            continue
        for rec in _heal_jsonl(path):
            if "replay_of" in rec:
                done.add((rec["replay_of"], rec["press"], str(rec["budget"])))
    return done


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


def _readout_path(root: Path, press: str, budget: int | str, seed: int) -> Path:
    # budget is int for compression cells, the literal "full" for full-cache
    # references (e.g. the FC-readout resampling control), matching bank naming.
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
    scale = bool(config.get("scale_authorized"))
    # Amendment P.4: opt-in per config so existing replay configs keep
    # emitting v1 profiles and nothing already published changes shape.
    extended_measures = bool(config.get("extended_measures"))
    if not (config.get("pilot") or scale) or config.get("mode") not in {
        "self_replay",
        "compression_replay",
        "readout",
    }:
        raise SystemExit("REFUSED: replay config needs pilot/scale_authorized + a mode")

    pilot_count = int(config["estimated_replay_count"])
    if not scale and pilot_count not in {1, 20, 80}:
        raise SystemExit("REFUSED: M4 pilot must contain 1, 20, or 80 conditions")
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
    base_runs = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    reference_root = base_runs / config["reference_run_name"]
    if not reference_root.is_dir():
        raise SystemExit(f"missing reference run: {reference_root}")
    reference_rows = _load_jsonl(reference_root)
    token_records = _load_token_records(reference_root)
    if len(reference_rows) != len(token_records):
        raise SystemExit("reference rows and token-ID records must be 1:1")
    if not scale and len(reference_rows) != 20:
        raise SystemExit("M4 pilot reference must contain exactly 20 rows")

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
    bench = str(config.get("benchmarks", ["math500"])[0])
    default_subset = (
        {"count": 5, "stratify_by": "level"} if bench == "math500" else None
    )
    problems = load_benchmark(
        bench,
        benchmarks_cfg[bench],
        subset=config.get("subset", {}).get(bench, default_subset),
        allow_download=False,
    )
    gold_by_pid = {problem.pid: problem.gold_answer for problem in problems}
    # Attention-score presses (H2O) need eager attention during replay/readout too, not
    # just free-run generation -- mirror run_grid so H2O/R-KV decomposition is correct.
    eager = any(press_needs_eager_attention(p) for p in config["presses"])
    model, tokenizer = load_model_and_tokenizer(
        model_cfg, allow_download=False, attn_implementation="eager" if eager else None
    )
    output_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    mode = config["mode"]
    completed = 0

    # Replay only the configured subset's problems and seeds (so a 250-problem FC
    # reference is not fully replayed when only a 100-problem cell is being decomposed).
    eligible_rows = [
        row
        for row in reference_rows
        if row["model"] == model_name
        and int(row["decoding"]["seed"]) in active_seeds
        and row["pid"] in gold_by_pid
    ]
    reference_limit = config.get("reference_limit")
    if reference_limit is not None:
        eligible_rows = eligible_rows[: int(reference_limit)]

    # The shard's real workload; the config's estimated_replay_count is only the
    # full-job sanity number used by the pilot guard above.
    arms = _condition_arms(config)
    budget_arms = [(b, a) for b in config["budgets"] for a in arms]
    expected = (
        len(eligible_rows) * len(config["presses"]) * len(config["budgets"]) * len(arms)
    )
    done_keys = _replay_done_keys(output_root, mode) if args.resume else set()
    skipped = 0
    if args.resume:
        print(f"RESUME: {len(done_keys)} conditions already complete in this run dir")

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
            for budget, arm in budget_arms:
                # Each restoration arm is its own cell, so arms never collide on
                # disk and --resume is per-arm.
                cell_name = press_name if arm is None else f"{press_name}.{arm}"
                if args.resume:
                    cond = (
                        (_reference_key(row), cell_name, str(budget))
                        if mode == "readout"
                        else (
                            f"{config['reference_run_name']}:{_reference_key(row)}",
                            cell_name,
                            str(budget),
                        )
                    )
                    if cond in done_keys:
                        skipped += 1
                        continue
                press = _build_condition_press(
                    press_name,
                    budget=budget,
                    config=config,
                    # Per-(problem, seed) RNG (decorrelate random eviction).
                    seed=press_seed(row["pid"], seed),
                    arm=arm,
                )
                backend = TransformersReplayBackend(model)
                runmeta = capture_runmeta(config["name"], args.config, seed)
                if mode == "readout":
                    import torch

                    boundary, boundary_kind = reasoning_boundary_token(
                        tokenizer, trace_ids, row["trace"]
                    )
                    if boundary is None:
                        # Undefined reasoning/answer boundary: exclude from
                        # readout-accuracy, never silently drop or mis-segment.
                        record = {
                            "schema_version": "readout-v1",
                            "reference_key": list(_reference_key(row)),
                            "condition": {"press": press_name, "budget": budget},
                            "boundary_token": None,
                            "boundary_kind": boundary_kind,
                            "readout_defined": False,
                            "generated_token_ids": [],
                            "generated_text": "",
                            "answer_raw": "",
                            "readout_correct": None,
                            "runmeta": runmeta,
                            "versions": row["versions"],
                        }
                    else:
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
                            "boundary_kind": boundary_kind,
                            "readout_defined": True,
                            "generated_token_ids": generated_ids,
                            "generated_text": generated_text,
                            "answer_raw": answer_raw,
                            "readout_correct": score_readout_answer(
                                gold_by_pid[row["pid"]], generated_text
                            ),
                            "runmeta": runmeta,
                            "versions": row["versions"],
                        }
                    path = _readout_path(
                        output_root,
                        press_name,
                        budget if budget == "full" else int(budget),
                        seed,
                    )
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
                            extended_measures=(
                                measures_from_logits if extended_measures else None
                            ),
                        )
                    if mode == "self_replay" and max(profile.estimate) > 1e-6:
                        raise SystemExit("self-replay TV exceeded 1e-6")
                    # The post-prune bound is variant-specific. A fixed-budget
                    # replay must land at or under `budget`; the relaxed
                    # restoration arm is EXPECTED to exceed it by exactly the
                    # reinserted count (that is what makes it the confounded
                    # reference arm); the ratio-matched press has no fixed budget
                    # at all, so the fixed check would be meaningless.
                    skip_budget_check = config.get("matched_ratio") is not None
                    allowance = (
                        int(getattr(press, "n_reinsert", 0)) if arm == "relaxed" else 0
                    )
                    if (
                        profile.compression_events
                        and budget != "full"
                        and not skip_budget_check
                    ):
                        if profile.max_after_prune is None or (
                            profile.max_after_prune > int(budget) + allowance
                        ):
                            raise SystemExit("post-prune cache exceeded target budget")
                        # Between prunes the cache holds the prefill (prompt) + one
                        # compression interval before the first prune, so the bound is
                        # max(prompt_len, budget)+interval -- NOT budget+interval. Long-
                        # prompt MATH problems (prompt > budget) tripped the old bound.
                        # Post-prune <= budget (above) still checks the press prunes.
                        # The relaxed arm's one-off reinsertion inflates the
                        # BETWEEN-prune peak too, not just the post-prune size:
                        # the cache reaches budget+n_reinsert and then accrues a
                        # further interval before the next prune. The smoke gate
                        # caught this bound being applied to only half the check.
                        prune_bound = (
                            max(len(prompt_ids), int(budget))
                            + int(config["compression_interval"])
                            + allowance
                        )
                        if (
                            profile.max_before_prune is None
                            or profile.max_before_prune > prune_bound
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
                        press_name=cell_name,
                        budget=budget,
                        pid_hash=pid_hash,
                        profile=profile,
                        # On resume an orphan profile can survive a timed stop that
                        # killed the process before the row was appended; overwrite it.
                        overwrite=args.resume,
                    )
                    if arm is not None:
                        # The instrument gate travels with the profile it gates.
                        # An arm that should have intervened and did not is
                        # recorded as such rather than left absent, so a reader
                        # cannot mistake silence for a passing gate.
                        report = getattr(press, "restoration_report", None)
                        if report is None:
                            report = {
                                "passed": arm == "none",
                                "arm": arm,
                                "intervened": False,
                                "note": (
                                    "no intervention was recorded; expected only "
                                    "for the 'none' arm"
                                ),
                            }
                        gate_path = profile_path.with_name(
                            profile_path.stem + ".restoration.json"
                        )
                        gate_path.write_text(
                            json.dumps(report, indent=2, sort_keys=True, default=str),
                            encoding="utf-8",
                        )
                    if config.get("matched_ratio") is not None:
                        # The trace IS the evidence that the ratio was held fixed
                        # and the horizon allowed to vary. Without it the control
                        # is an assertion.
                        trace_path = profile_path.with_name(
                            profile_path.stem + ".ratio_trace.json"
                        )
                        trace_path.write_text(
                            json.dumps(
                                {
                                    "matched_ratio": float(config["matched_ratio"]),
                                    "min_budget": int(
                                        config.get("matched_min_budget", 32)
                                    ),
                                    "trace": getattr(press, "ratio_trace", []),
                                },
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    logical_tokens = len(prompt_ids) + len(trace_ids)
                    ratio = (
                        1.0
                        if budget == "full"
                        else min(1.0, int(budget) / logical_tokens)
                    )
                    replay_row = build_replay_row(
                        row,
                        press_name=cell_name,
                        budget=budget,
                        compression_ratio=ratio,
                        replay_of=f"{config['reference_run_name']}:{_reference_key(row)}",
                        tv_profile_path=str(profile_path.relative_to(output_root)),
                        profile=profile,
                        runmeta=runmeta,
                    )
                    append_result(output_root, replay_row)
                completed += 1

    if completed + skipped != expected:
        raise SystemExit(
            f"expected {expected} conditions, completed {completed} + "
            f"resumed {skipped} = {completed + skipped}"
        )
    print(
        json.dumps(
            {"mode": mode, "completed": completed, "resumed": skipped},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

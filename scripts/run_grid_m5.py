#!/usr/bin/env python3
"""Meeting 5 protected-policy free runs with production identity (review of
6b409001, R2; prepared 2026-10-14).

Replaces the identity-less adapter path for NEW M5 cells (``rkv_pp_v1``,
``h2o_style_pp_v1``, ``uniform_fresh_pp_v1``). ``scripts/run_grid.py`` is not
modified: its bytes are generation identity for finished cells, and the
approved P3 pilot still uses ``run_grid_protected.py``. This runner reuses the
same package builders and generation call, and adds what the review asked for
(``kvcache.m5_runs``):

1. the expected identity is computed, and every existing row and sidecar in
   the namespace is checked against it, BEFORE the model loads -- a changed
   family implementation, config, model revision or reference input refuses
   the resume;
2. every new row and sidecar carries that identity, the actual prompt-input
   hash (checked against the frozen manifest before writing) and the per-row
   capacity/retention block;
3. the family's own gate runs first (held flag + validated smoke PASS).

    python scripts/run_grid_m5.py \\
        --config configs/experiments/grid_math500_rkvpp256_m5.yaml --dry-run
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

from run_grid import _grid_done_keys, _parser, _yaml

from kvcache.bank import append_result, build_result_row
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.generation import generate_compressed
from kvcache.m5_runs import (
    expected_identity,
    family_kind,
    frozen_prompt_hash,
    refuse_foreign_namespace,
    retention_block,
)
from kvcache.m5_runs import design_keys as m5_design_keys
from kvcache.models import (
    load_model_and_tokenizer,
    render_prompt,
    tokenize_rendered_prompt,
)
from kvcache.presses import press_seed
from kvcache.provenance import inputs_sha256, load_input_manifest
from kvcache.read_validation import frozen_pids
from kvcache.runmeta import capture_runmeta
from kvcache.token_ids import append_token_ids, build_token_ids_record


def family_api(kind: str) -> tuple[Any, Any, Any]:
    """(press builder, eager selector, config gate) of one M5 family."""
    if kind == "rkv_pp":
        from kvcache.presses import rkv_prompt_protect as m

        return (
            m.build_rkv_prompt_protected_press,
            m.needs_eager_attention,
            (lambda cfg, **kw: m.require_rkv_pp_config(cfg, **kw)),
        )
    from kvcache.presses import protected_selectors as m

    return (
        m.build_protected_selector_press,
        m.needs_eager_attention,
        (lambda cfg, **kw: m.require_protected_config(cfg, **kw)),
    )


def _model_shape(model: Any) -> dict[str, int]:
    c = model.config
    heads = int(c.num_attention_heads)
    return {
        "n_layers": int(c.num_hidden_layers),
        "n_kv_heads": int(getattr(c, "num_key_value_heads", heads)),
        "head_dim": int(getattr(c, "head_dim", None) or c.hidden_size // heads),
        "dtype_bytes": int(model.dtype.itemsize),
    }


def main() -> int:
    args = _parser().parse_args()
    config = load_config(args.config)
    plan(config, args.config, "grid_m5")
    kind = family_kind(list(config["presses"]))
    build, needs_eager, require = family_api(kind)
    ident = expected_identity(config)
    pids = frozen_pids(int(config["subset"]["math500"]), root=PROJECT_ROOT)
    keys = m5_design_keys(config, pids)
    print(json.dumps({"family_kind": kind, "identity": ident}, sort_keys=True))
    require(
        config,
        dry_run=bool(args.dry_run),
        runs_root=os.environ.get("KVCACHE_BASE_RUNS_ROOT"),
    )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    if not config.get("scale_authorized"):
        raise SystemExit("REFUSED: M5 grid config must set scale_authorized: true")
    if list(config["benchmarks"]) != ["math500"]:
        raise SystemExit("REFUSED: M5 grids are registered for math500 only")

    runs_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    manifest = load_input_manifest(path=config.get("input_manifest"))
    checked = refuse_foreign_namespace(
        runs_root, ident=ident, manifest=manifest, keys=keys
    )
    print(f"IDENTITY: {checked} existing records match this run")

    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    sampling_cfg = _yaml(PROJECT_ROOT / "configs/sampling.yaml")
    model_name = config["models"][0]
    model_cfg = models_cfg[model_name]
    eager = any(needs_eager(p) for p in config["presses"])
    model, tokenizer = load_model_and_tokenizer(
        model_cfg, attn_implementation="eager" if eager else None
    )
    shape = _model_shape(model)
    store_token_ids = bool(config.get("store_token_ids"))
    seeds = [int(s) for s in config["seeds"]]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in shard]
        if not seeds:
            raise SystemExit(f"--seeds {args.seeds} disjoint from the config seeds")
    done = _grid_done_keys(runs_root, store_token_ids) if args.resume else set()
    counts = {"rows": 0, "truncated": 0, "skipped_resume": len(done)}
    interval = int(config["compression_interval"])
    max_new = int(config["max_new_tokens"]["math500"])
    problems = load_benchmark(
        "math500",
        benchmarks_cfg["math500"],
        subset=config["subset"]["math500"],
        allow_download=False,
    )
    if [p.pid for p in problems] != pids:
        raise SystemExit("REFUSED: loaded problems differ from the frozen pid list")
    for press_name in config["presses"]:
        for budget in config["budgets"]:
            for seed in seeds:
                runmeta = capture_runmeta(config["name"], args.config, seed)
                for problem in problems:
                    key = ("math500", problem.pid, press_name, str(budget), seed)
                    if key in done:
                        continue
                    press = build(
                        press_name,
                        budget=int(budget),
                        compression_interval=interval,
                        seed=press_seed(problem.pid, seed),
                    )
                    generation = generate_compressed(
                        model,
                        tokenizer,
                        problem,
                        model_cfg,
                        sampling_cfg,
                        press=press,
                        seed=seed,
                        max_new_tokens=max_new,
                    )
                    got = inputs_sha256(generation.prompt_token_ids)
                    if got != frozen_prompt_hash(manifest, problem.pid):
                        raise SystemExit(
                            f"REFUSED: prompt inputs of {problem.pid} differ from "
                            "the frozen manifest; nothing written"
                        )
                    prompt_tokens = len(
                        tokenize_rendered_prompt(
                            tokenizer,
                            render_prompt(tokenizer, problem.question, model_cfg),
                        )["input_ids"]
                    )
                    row = build_result_row(
                        bench="math500",
                        pid=problem.pid,
                        model_name=model_name,
                        seed=seed,
                        max_new_tokens=max_new,
                        sampling_cfg=sampling_cfg,
                        generation=generation,
                        runmeta=runmeta,
                        press=press_name,
                        budget=int(budget),
                        compression_ratio=min(
                            1.0, int(budget) / (prompt_tokens + generation.n_tokens)
                        ),
                    )
                    row["provenance"] = {**ident, "inputs_sha256": got}
                    row["m5"] = retention_block(
                        press,
                        family=press_name,
                        nominal_budget=int(budget),
                        interval=interval,
                        **shape,
                    )
                    output = append_result(runs_root, row)
                    if store_token_ids:
                        rec = build_token_ids_record(
                            result_row=row,
                            prompt_token_ids=generation.prompt_token_ids,
                            trace_token_ids=generation.trace_token_ids,
                            prompt_text=generation.prompt,
                            trace_text=generation.trace,
                            tokenizer=tokenizer,
                            tokenizer_revision=str(model_cfg["revision"]),
                        )
                        rec["provenance"] = row["provenance"]
                        append_token_ids(runs_root, rec)
                    counts["rows"] += 1
                    counts["truncated"] += int(row["truncated"])
                    print(f"wrote {output}: math500/{problem.pid}/s{seed}")
    if counts["rows"]:
        counts["truncation_rate"] = counts["truncated"] / counts["rows"]
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Information-channel probe runner (Meeting 5, M5-C; review of 6b409001, R3;
prepared 2026-10-14 -- HELD, not approved).

Builds the controlled target/nuisance task (``kvcache.information_tasks``) on
the PINNED tokenizer, runs every (S, N) variant of every block under every
policy through ``kvcache.information_probe`` (decode-path body with eviction
on; two independent cache clones per query with eviction off; full/full,
clone-agreement and no-readout-eviction controls), and writes production rows
plus the frozen design the reader enforces
(``scripts/report_information_probe.py``).

Execution (only after its own approval removes ``held``): both protected
press families' smoke PASS records are validated first, the encodings are
re-checked on the pinned tokenizer, filler comes from MATH-500 reserve traces,
``design.json`` is written (or must match), a resume into rows of another
identity is refused, and each block's rows are written atomically under
``$REMOTE_RUNS_ROOT/readout/``. Read with ``scripts/report_information_probe.py
--design design.json --calibration-criteria <this config>``.

    python scripts/run_information_probe.py --dry-run
    KVCACHE_REMOTE=1 python scripts/run_information_probe.py --check-tokenizer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache import information_probe as ip
from kvcache import information_tasks as it
from kvcache.provenance import implementation_id

CONFIG = "configs/meeting5/info_calibration.yaml"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def spec_of(cfg: dict[str, Any]) -> it.TaskSpec:
    return it.TaskSpec(**cfg["spec"])


def design(cfg: dict[str, Any], block_ids: list[str]) -> dict[str, Any]:
    """The frozen design the reader enforces, with its expected provenance."""
    models = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    rev = str(models[cfg["models"][0]]["revision"])
    layout = {"spec": asdict(spec_of(cfg)), "aliases": cfg["aliases"]}
    return {
        "mode": "production",
        "policies": list(cfg["policies"]),
        "reference_policy": "full",
        "contexts": block_ids,
        "eviction_seeds": list(cfg["eviction_seeds"]),
        "queries": list(cfg["queries"]),
        "paired_contrasts": cfg.get("paired_contrasts", []),
        "expected_provenance": {
            "implementation_id": implementation_id(ip.INFO_KIND),
            "model_revision": rev,
            "tokenizer_revision": rev,
            "answer_alphabet_sha256": ip.sha256_json(cfg["values"]),
            "task_layout_sha256": ip.sha256_json(layout),
            "design_sha256": ip.sha256_json(cfg),
        },
    }


def counts(cfg: dict[str, Any]) -> dict[str, int]:
    n_var = int(cfg["n_blocks"]) * it.K * it.K
    prefixes = n_var * len(cfg["policies"]) * len(cfg["eviction_seeds"])
    return {
        "variants": n_var,
        "prefix_executions": prefixes,
        "readouts": prefixes * len(cfg["queries"]) * 2,  # two clones per query
        "full_full_controls": int(cfg["n_blocks"]) * 2,
    }


def check_tokenizer(cfg: dict[str, Any]) -> list[str]:
    from kvcache.models import load_tokenizer

    models = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    tok = load_tokenizer(models[cfg["models"][0]])
    return it.check_encodings(
        lambda t: tok.encode(t, add_special_tokens=False),
        cfg["values"],
        cfg["aliases"],
        spec_of(cfg),
    )


def load_passages(cfg: dict[str, Any], base_runs: Path) -> list[it.Passage]:
    """Filler from full-cache reasoning traces of the MATH-500 RESERVE problems
    (outside the discovery cohort and the confirmation set), seed per config;
    each passage keeps the hash of the exact token ids it was cut from."""
    from kvcache.provenance import inputs_sha256

    f = cfg["filler"]
    manifest = json.loads((PROJECT_ROOT / f["pids_from"]).read_text())
    node: Any = manifest
    for k in f["pids_key"]:
        node = node[k]
    reserve = set(node)
    seed, n, length = int(f["seed"]), int(f["n_passages"]), int(f["passage_tokens"])
    out = []
    side = base_runs / f["reference_run"] / "token_ids"
    for path in sorted(side.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["pid"] in reserve and int(rec["seed"]) == seed:
                ids = tuple(int(t) for t in rec["trace_token_ids"][:length])
                if len(ids) == length:
                    out.append(it.Passage(inputs_sha256(ids), ids))
    out.sort(key=lambda p: p.source)
    if len(out) < n:
        raise SystemExit(f"REFUSED: only {len(out)} filler passages, need {n}")
    return out[:n]


def make_backend(model: Any) -> Any:
    return ip.TransformersBackend(model)


def policy_factories(model: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Press factory per configured policy (seed -> PressHandle)."""
    from kvcache.presses.prompt_protect import build_prompt_protected_press
    from kvcache.presses.protected_selectors import build_protected_selector_press
    from kvcache.presses.rkv_prompt_protect import build_rkv_prompt_protected_press

    budget, interval = int(cfg["budget"]), int(cfg["compression_interval"])
    builders = {
        "random_pp": build_prompt_protected_press,
        "streamingllm_pp": build_prompt_protected_press,
        "h2o_style_pp_v1": build_protected_selector_press,
        "rkv_pp_v1": build_rkv_prompt_protected_press,
    }

    def factory(name: str) -> Any:
        if name == "full":
            return lambda seed: ip.NoPress()
        build = builders[name]
        return lambda seed: ip.KVPressHandle(
            build(
                name,
                budget=budget,
                compression_interval=interval,
                seed=int(seed),
                instrumented=True,
            ),
            model,
        )

    return {name: factory(name) for name in cfg["policies"]}


def smoke_gates(cfg: dict[str, Any], base_runs: Path) -> list[str]:
    """The probe uses protected selector and R-KV presses: both families'
    real-model smokes must have VALID PASS records first."""
    from kvcache.presses import protected_selectors as ps
    from kvcache.presses import rkv_prompt_protect as rp

    problems = []
    for mod, path in (
        (ps, "configs/experiments/protsel_smoke_wp4.yaml"),
        (rp, "configs/experiments/rkvpp_smoke_m5.yaml"),
    ):
        ok, why = mod.smoke_record_ok(base_runs, _yaml(PROJECT_ROOT / path))
        if not ok:
            problems.append(why)
    return problems


def _rows_in(out_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for f in sorted((out_dir / "readout").glob("*.jsonl"))
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def execute(cfg: dict[str, Any], *, out_dir: Path, base_runs: Path) -> dict:
    """Run the calibration (or resume it); every block is written atomically."""
    from kvcache.models import load_model_and_tokenizer, load_tokenizer
    from kvcache.provenance import write_json_atomic

    models = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    model_cfg = models[cfg["models"][0]]
    gates = smoke_gates(cfg, base_runs)
    if gates:
        raise SystemExit(f"REFUSED: smoke gate: {gates[0]}")
    tok = load_tokenizer(model_cfg)

    def encode(t: str) -> list[int]:
        return list(tok.encode(t, add_special_tokens=False))

    spec = spec_of(cfg)
    problems = it.check_encodings(encode, cfg["values"], cfg["aliases"], spec)
    if problems:
        raise SystemExit(f"REFUSED: pinned-tokenizer encoding check: {problems}")
    passages = load_passages(cfg, base_runs)
    blocks = it.make_blocks(
        int(cfg["n_blocks"]),
        design_seed=int(cfg["design_seed"]),
        aliases=cfg["aliases"],
        passages=passages,
        encode=encode,
        values=cfg["values"],
        spec=spec,
    )
    d = design(cfg, [b.block_id for b in blocks])
    dpath = out_dir / "design.json"
    if dpath.exists() and json.loads(dpath.read_text()) != d:
        raise SystemExit("REFUSED: an existing design.json differs; new namespace")
    write_json_atomic(dpath, d)
    prov = d["expected_provenance"]
    existing = _rows_in(out_dir)
    bad = ip.resume_problems(existing, prov)
    if bad:
        raise SystemExit(f"REFUSED: resume into other provenance: {bad[0]}")
    done = {r["context_id"] for r in existing}
    model, _ = load_model_and_tokenizer(model_cfg, attn_implementation="eager")
    backend = make_backend(model)
    factories = policy_factories(model, cfg)
    symbol_ids = [encode(" " + v)[0] for v in cfg["values"]]
    written = 0
    for block in blocks:
        if block.block_id in done:
            continue
        variants = it.all_variants(
            block, passages=passages, encode=encode, values=cfg["values"], spec=spec
        )
        rows = ip.run_context(
            backend,
            variants,
            policies=factories,
            eviction_seeds=cfg["eviction_seeds"],
            symbol_ids=symbol_ids,
            provenance=prov,
            queries=cfg["queries"],
        )
        path = (
            out_dir / "readout" / f"{block.block_id}-s{cfg['eviction_seeds'][0]}.jsonl"
        )
        tmp = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        os.replace(tmp, path)  # a block is all-or-nothing
        written += len(rows)
        print(f"wrote {path.name}: {len(rows)} rows", flush=True)
    return {"rows_written": written, "blocks": len(blocks), "skipped": len(done)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-tokenizer", action="store_true")
    ap.add_argument("--i-understand", action="store_true")
    # accepted so the nightly launcher can run this like any other stage
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    cfg = _yaml(PROJECT_ROOT / a.config)
    print(json.dumps({"config": a.config, "held": cfg.get("held"), **counts(cfg)}))
    if a.dry_run:
        d = design(
            cfg, [f"b{cfg['design_seed']}-{i:04d}" for i in range(cfg["n_blocks"])]
        )
        print(json.dumps(d, indent=1, sort_keys=True))
        print(json.dumps({"calibration_criteria": cfg["calibration_criteria"]}))
        print("DRY RUN: design, counts and criteria only; no tokenizer, model or GPU")
        return 0
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: tokenizer and model work is remote-only")
    if a.check_tokenizer:
        problems = check_tokenizer(cfg)
        print(json.dumps({"tokenizer_problems": problems}))
        return 2 if problems else 0
    if cfg.get("held", True):
        raise SystemExit(
            "REFUSED: the information calibration is HELD (separate approval, "
            "proposed cap 4 GPU-h)"
        )
    if not a.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    out = execute(
        cfg,
        out_dir=Path(os.environ["REMOTE_RUNS_ROOT"]),
        base_runs=Path(os.environ["KVCACHE_BASE_RUNS_ROOT"]),
    )
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

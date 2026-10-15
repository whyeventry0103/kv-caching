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


MANIFEST_NAME = "input_manifest.json"


def design(
    cfg: dict[str, Any],
    block_ids: list[str],
    *,
    manifest_sha256: str | None,
    schedule: dict[str, Any] | None,
) -> dict[str, Any]:
    """The frozen design the reader enforces, with its expected provenance.

    Review of 5616630: it binds the ACTUAL selected inputs (the input
    manifest's hash, R2), the measurement schedule the calibration fields are
    validated against (R1) and the calibration criteria with their fingerprint
    (R1.4), so a reader cannot substitute another criteria file. ``None``
    values appear only in the dry-run preview (no tokenizer, no filler)."""
    models = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    rev = str(models[cfg["models"][0]]["revision"])
    layout = {"spec": asdict(spec_of(cfg)), "aliases": cfg["aliases"]}
    crit = cfg["calibration_criteria"]
    return {
        "mode": "production",
        "policies": list(cfg["policies"]),
        "reference_policy": "full",
        "contexts": block_ids,
        "eviction_seeds": list(cfg["eviction_seeds"]),
        "queries": list(cfg["queries"]),
        "paired_contrasts": cfg.get("paired_contrasts", []),
        "measurement_schedule": schedule,
        "input_manifest": MANIFEST_NAME,
        "calibration_criteria": crit,
        "calibration_criteria_sha256": ip.sha256_json(crit),
        "expected_provenance": {
            "implementation_id": implementation_id(ip.INFO_KIND),
            "model_revision": rev,
            "tokenizer_revision": rev,
            "answer_alphabet_sha256": ip.sha256_json(cfg["values"]),
            "task_layout_sha256": ip.sha256_json(layout),
            "design_sha256": ip.sha256_json(cfg),
            "input_manifest_sha256": manifest_sha256,
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
    out = eligible_passages(cfg, base_runs)
    n = int(cfg["filler"]["n_passages"])
    if len(out) < n:
        raise SystemExit(f"REFUSED: only {len(out)} filler passages, need {n}")
    return out[:n]


def eligible_passages(cfg: dict[str, Any], base_runs: Path) -> list[it.Passage]:
    """Every reserve trace long enough to cut one passage from, sorted."""
    from kvcache.provenance import inputs_sha256

    f = cfg["filler"]
    manifest = json.loads((PROJECT_ROOT / f["pids_from"]).read_text())
    node: Any = manifest
    for k in f["pids_key"]:
        node = node[k]
    reserve = set(node)
    seed, length = int(f["seed"]), int(f["passage_tokens"])
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
    return out


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


def prepare_inputs(cfg: dict[str, Any], base_runs: Path, encode: Any) -> dict[str, Any]:
    """Everything a run consumes, materialized WITHOUT model weights: filler
    passages, blocks, every rendered variant, the input manifest and its hash,
    the measurement schedule and the frozen design (review of 5616630, R2)."""
    spec = spec_of(cfg)
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
    variants = {
        b.block_id: it.all_variants(
            b, passages=passages, encode=encode, values=cfg["values"], spec=spec
        )
        for b in blocks
    }
    symbol_ids = [encode(" " + v)[0] for v in cfg["values"]]
    manifest = it.input_manifest(
        blocks, variants, passages=passages, symbol_ids=symbol_ids
    )
    schedule = ip.measurement_schedule(variants, cfg["queries"])
    d = design(
        cfg,
        [b.block_id for b in blocks],
        manifest_sha256=ip.sha256_json(manifest),
        schedule=schedule,
    )
    return {
        "passages": passages,
        "blocks": blocks,
        "variants": variants,
        "symbol_ids": symbol_ids,
        "manifest": manifest,
        "design": d,
    }


def resume_refusals(
    out_dir: Path, d: dict[str, Any], manifest: dict[str, Any]
) -> tuple[set[str], list[str]]:
    """(finished blocks, reasons to refuse) for a run directory, checked
    BEFORE any model load or row write: the stored design and input manifest
    must equal the ones just materialized from the actual inputs, every stored
    row must carry the same identity, and every stored block must be
    complete."""
    reasons = []
    dpath, mpath = out_dir / "design.json", out_dir / MANIFEST_NAME
    if dpath.exists() and json.loads(dpath.read_text()) != d:
        reasons.append(
            "an existing design.json differs (config or ACTUAL selected inputs "
            "changed); use a new namespace"
        )
    if mpath.exists() and ip.sha256_json(json.loads(mpath.read_text())) != (
        ip.sha256_json(manifest)
    ):
        reasons.append("an existing input manifest differs from the selected inputs")
    existing = _rows_in(out_dir)
    if existing and not (dpath.exists() and mpath.exists()):
        reasons.append("stored rows without their design.json and input manifest")
    reasons += [
        f"resume into other provenance: {b}"
        for b in ip.resume_problems(existing, d["expected_provenance"])[:3]
    ]
    done, incomplete = ip.finished_blocks(
        existing,
        contexts=d["contexts"],
        policies=d["policies"],
        eviction_seeds=d["eviction_seeds"],
        queries=d["queries"],
    )
    return done, reasons + incomplete


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
    inputs = prepare_inputs(cfg, base_runs, encode)
    d, manifest = inputs["design"], inputs["manifest"]
    done, refusals = resume_refusals(out_dir, d, manifest)
    if refusals:
        raise SystemExit(f"REFUSED: {refusals[0]}")
    write_json_atomic(out_dir / MANIFEST_NAME, manifest)
    write_json_atomic(out_dir / "design.json", d)
    prov = d["expected_provenance"]
    model, _ = load_model_and_tokenizer(model_cfg, attn_implementation="eager")
    backend = make_backend(model)
    factories = policy_factories(model, cfg)
    symbol_ids = inputs["symbol_ids"]
    written = 0
    for block in inputs["blocks"]:
        if block.block_id in done:
            continue
        variants = inputs["variants"][block.block_id]
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
    return {
        "rows_written": written,
        "blocks": len(inputs["blocks"]),
        "skipped": len(done),
    }


def check_inputs(cfg: dict[str, Any], base_runs: Path) -> dict[str, Any]:
    """Feasibility WITHOUT model weights (review of 5616630, item 5): the real
    reserve filler exists in sufficient quantity, and the pinned tokenizer
    renders the frozen task (encodings, blocks, every variant, manifest)."""
    from kvcache.models import load_tokenizer

    models = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    tok = load_tokenizer(models[cfg["models"][0]])

    def encode(t: str) -> list[int]:
        return list(tok.encode(t, add_special_tokens=False))

    report: dict[str, Any] = {
        "eligible_passages": len(eligible_passages(cfg, base_runs)),
        "needed_passages": int(cfg["filler"]["n_passages"]),
        "tokenizer_problems": it.check_encodings(
            encode, cfg["values"], cfg["aliases"], spec_of(cfg)
        ),
    }
    if report["tokenizer_problems"] or (
        report["eligible_passages"] < report["needed_passages"]
    ):
        return report
    inputs = prepare_inputs(cfg, base_runs, encode)
    m = inputs["manifest"]
    report.update(
        blocks=len(m["blocks"]),
        variants=sum(len(v) for v in m["variants"].values()),
        skipped_sources=sum(len(b["skipped_sources"]) for b in m["blocks"]),
        body_tokens={len(v.body_ids) for vs in inputs["variants"].values() for v in vs}
        == {spec_of(cfg).body_tokens},
        fact_spans=inputs["design"]["measurement_schedule"]["fact_spans"],
        input_manifest_sha256=inputs["design"]["expected_provenance"][
            "input_manifest_sha256"
        ],
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-tokenizer", action="store_true")
    ap.add_argument(
        "--check-inputs",
        action="store_true",
        help="reserve filler + pinned tokenizer render the frozen task; no weights",
    )
    ap.add_argument("--i-understand", action="store_true")
    # accepted so the nightly launcher can run this like any other stage
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    cfg = _yaml(PROJECT_ROOT / a.config)
    print(json.dumps({"config": a.config, "held": cfg.get("held"), **counts(cfg)}))
    if a.dry_run:
        d = design(
            cfg,
            [f"b{cfg['design_seed']}-{i:04d}" for i in range(cfg["n_blocks"])],
            manifest_sha256=None,
            schedule=None,
        )
        print(json.dumps(d, indent=1, sort_keys=True))
        print(
            "DRY RUN: design, counts and criteria only; the input manifest and "
            "measurement schedule (null above) are materialized at execution "
            "from the pinned tokenizer and the actual filler; no model or GPU"
        )
        return 0
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: tokenizer and model work is remote-only")
    if a.check_tokenizer:
        problems = check_tokenizer(cfg)
        print(json.dumps({"tokenizer_problems": problems}))
        return 2 if problems else 0
    if a.check_inputs:
        rep = check_inputs(cfg, Path(os.environ["KVCACHE_BASE_RUNS_ROOT"]))
        print(json.dumps(rep, indent=1, sort_keys=True))
        ok = (
            not rep["tokenizer_problems"]
            and rep["eligible_passages"] >= rep["needed_passages"]
            and rep.get("body_tokens") is True
        )
        return 0 if ok else 2
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

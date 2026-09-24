#!/usr/bin/env python3
"""Meeting #3 Tasks 2 + 3 capture (R.4, R.6, R.9): one replay, two instruments.

For every reference trace of the Q.2 subset (FC bank, sharded by seed): the
full-cache branch is replayed once (``kvcache.capture.record_full_branch``),
then each (policy, budget) condition replays only its compressed branch
against it (``replay_condition``), producing the Q.3 attention channels (M,
M_event, Delta, GQA collateral) and the Q.6 latent split (omega) in the same
pass. The model is loaded with eager attention for every policy (R.4).

Outputs, per condition: one row in ``<run>/math500/<model>/<press>-<budget>-
s<seed>.jsonl`` (``capture-v1``: the compact attn-profile-v2 and latent-proj-v1
summaries, R.9) and one ``<run>/latent/<press>-<budget>-s<seed>/<pid-hash>.npz``
with the (||d_par||, ||d_perp||) norms at the 256 sampled positions (float16).

``smoke: true`` runs the configured traces then writes
``<base runs>/capture-smoke-PASS.json`` only if every registered control holds:
full/full exactly zero, layer-0 Delta within the algebra tolerance, no
non-finite values, at least one event per policy. Otherwise it writes the FAIL
report and exits 1, which blocks every capture cell behind it in the queue.

    python scripts/run_capture.py \\
        --config configs/experiments/capture_math500_b256_r9.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_basis_fit import BASIS_HEADER, BASIS_NPZ, sha256_file
from run_replay import _heal_jsonl, _load_jsonl, _load_token_records, _reference_key

from kvcache.capture import LAYER0_DELTA_TOL
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    same_identity,
    write_gate,
)
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.runmeta import capture_runmeta

SCHEMA = "capture-v1"
SMOKE_PASS = "capture-smoke-PASS.json"
SMOKE_FAIL = "capture-smoke-FAIL.json"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seeds", default="")
    return p


def _round(x: Any) -> Any:
    """6 significant digits for floats, recursively: halves the row size."""
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def _persisted(out_root: Path, ident: dict[str, Any]) -> list[dict[str, Any]]:
    """Every row already in this run namespace, refusing foreign ones.

    A row whose identity (implementation, config, model, reference, basis)
    differs from this run's cannot be resumed into it: the run stops and asks
    for a new namespace instead of silently mixing experiments.
    """
    rows: list[dict[str, Any]] = []
    base = out_root / "math500"
    for path in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        for rec in _heal_jsonl(path):
            if not same_identity((rec.get("provenance") or {}), ident):
                raise SystemExit(
                    f"REFUSED: {path} holds a row from a different experiment "
                    f"identity; start a new run namespace (old rows are kept)"
                )
            if rec["press"] != "fullcache":
                check_inputs(
                    load_input_manifest(),
                    "capture",
                    rec["pid"],
                    int(rec["seed"]),
                    rec["provenance"].get("inputs_sha256"),
                )
            rows.append(rec)
    return rows


def _key(rec: dict[str, Any]) -> tuple[str, int, str, str]:
    return (rec["pid"], int(rec["seed"]), rec["press"], str(rec["budget"]))


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "capture")
    conditions = [(p, b) for p in cfg["presses"] for b in cfg["budgets"]]
    print(
        json.dumps(
            {
                "conditions": conditions,
                "fullfull_control": bool(cfg.get("fullfull_control")),
                "smoke": bool(cfg.get("smoke")),
                "basis_run": cfg.get("basis_run"),
                "tau": cfg["tau"],
                "reference_limit": cfg.get("reference_limit"),
            }
        )
    )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    import numpy as np
    import torch

    from kvcache.capture import record_full_branch, replay_condition
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    out_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    interval, tau = int(cfg["compression_interval"]), float(cfg["tau"])

    # Frozen basis: the bytes must match the hash recorded when it was frozen.
    basis = None
    basis_sha = None
    basis_header: dict[str, Any] = {}
    if cfg.get("basis_run"):
        bdir = base / cfg["basis_run"] / "basis"
        header = json.loads((bdir / BASIS_HEADER).read_text(encoding="utf-8"))
        basis_sha = sha256_file(bdir / BASIS_NPZ)
        if basis_sha != header["npz_sha256"]:
            raise SystemExit("REFUSED: basis bytes do not match their frozen hash")
        if header.get("reference_run_name") != cfg["reference_run_name"]:
            raise SystemExit("REFUSED: basis was fitted on a different reference run")
        arr = np.load(bdir / BASIS_NPZ)
        basis = {"scale": arr["scale"], "basis": arr["basis"]}
        basis_header = header

    ref_root = base / cfg["reference_run_name"]
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    problems = load_benchmark(
        "math500",
        benchmarks_cfg["math500"],
        subset=cfg["subset"]["math500"],
        allow_download=False,
    )
    pids = {p.pid for p in problems}
    seeds = [int(s) for s in cfg["seeds"]]
    if args.seeds.strip():
        shard = {int(s) for s in args.seeds.split(",")}
        seeds = [s for s in seeds if s in shard]
    model_name = cfg["models"][0]
    eligible = sorted(
        (
            r
            for r in rows
            if r["pid"] in pids
            and int(r["decoding"]["seed"]) in seeds
            and r["model"] == model_name
        ),
        key=lambda r: (int(r["decoding"]["seed"]), r["pid"]),
    )
    if cfg.get("reference_limit") is not None:
        eligible = eligible[: int(cfg["reference_limit"])]
    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    model_cfg = models_cfg[model_name]
    if basis is not None and basis_header.get("model_revision") != str(
        model_cfg["revision"]
    ):
        raise SystemExit("REFUSED: basis was fitted on a different model revision")
    ident = identity(
        "capture",
        cfg,
        model_revision=str(model_cfg["revision"]),
        reference_run=cfg["reference_run_name"],
        basis_npz_sha256=basis_sha,
    )
    manifest = load_input_manifest()
    persisted = _persisted(out_root, ident)
    done = {_key(r) for r in persisted} if args.resume else set()
    if not args.resume and persisted:
        raise SystemExit("REFUSED: run namespace not empty; pass --resume")
    model, _tok = load_model_and_tokenizer(
        model_cfg, allow_download=False, attn_implementation="eager"
    )
    if basis is not None:
        basis = {
            k: torch.tensor(v, device=model.device, dtype=torch.float32)
            for k, v in basis.items()
        }
    all_conds = list(conditions)
    if cfg.get("fullfull_control"):
        all_conds = [("fullcache", "full"), *all_conds]
    summaries: list[dict[str, Any]] = []
    for r in eligible:
        seed = int(r["decoding"]["seed"])
        todo = [c for c in all_conds if (r["pid"], seed, c[0], str(c[1])) not in done]
        if not todo:
            continue
        rec = tokens[_reference_key(r)]
        prompt, trace = rec["prompt_token_ids"], rec["trace_token_ids"]
        check_inputs(manifest, "capture", r["pid"], seed, inputs_sha256(prompt, trace))
        t0 = time.monotonic()
        full = record_full_branch(model, prompt, trace, interval=interval)
        full_s = time.monotonic() - t0
        runmeta = capture_runmeta(cfg["name"], args.config, seed)
        for press_name, budget in todo:
            t1 = time.monotonic()
            press = (
                None
                if press_name == "fullcache"
                else build_press(
                    press_name,
                    budget=int(budget),
                    compression_interval=interval,
                    seed=press_seed(r["pid"], seed),
                    instrumented=True,
                )
            )
            out = replay_condition(
                model,
                press,
                full,
                prompt,
                trace,
                budget=None if press is None else int(budget),
                interval=interval,
                tau=tau,
                basis=basis if press is not None else None,
                trigger_step=int(cfg.get("trigger_step", 130)),
            )
            sampled = out.pop("latent_sampled", None)
            pid_hash = hashlib.sha256(r["pid"].encode()).hexdigest()[:16]
            if sampled is not None:
                npz = (
                    out_root
                    / "latent"
                    / f"{press_name}-{budget}-s{seed}"
                    / f"{pid_hash}.npz"
                )
                npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    npz,
                    par=np.asarray(sampled["par"], dtype=np.float16),
                    perp=np.asarray(sampled["perp"], dtype=np.float16),
                    positions=np.asarray(out["latent"]["sampled_positions"]),
                )
            row = {
                "schema_version": SCHEMA,
                "pid": r["pid"],
                "seed": seed,
                "press": press_name,
                "budget": budget,
                "reference_key": list(_reference_key(r)),
                "n_trace_tokens": len(trace),
                "prompt_tokens": len(prompt),
                "compression_interval": interval,
                "tau": tau,
                "basis_npz_sha256": basis_sha,
                "wallclock_s": time.monotonic() - t1,
                "full_branch_s": full_s,
                "runmeta": {
                    k: runmeta.get(k)
                    for k in ("commit", "source_tree_hash", "gpu", "config_hash")
                },
                "provenance": {**ident, "inputs_sha256": inputs_sha256(prompt, trace)},
                **_round(out),
            }
            path = (
                out_root
                / "math500"
                / model_name
                / f"{press_name}-{budget}-s{seed}.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            summaries.append(row)
            print(
                f"{r['pid']} s{seed} {press_name}-{budget}: events={row['n_events']} "
                f"M={row['M_mean']} Delta={row['Delta_mean']} "
                f"coll={row['collateral_pooled']} "
                f"omega={(row.get('latent') or {}).get('omega_trace')} "
                f"l0={row['checks']['layer0_delta_max']:.2e} "
                f"{row['wallclock_s']:.0f}s",
                flush=True,
            )
        del full
        torch.cuda.empty_cache()

    if cfg.get("smoke"):
        # Evidence is rebuilt from EVERY persisted row of this identity, not
        # only this process's rows: a restart after the last row but before
        # the marker must still see the whole smoke.
        rows = _persisted(out_root, ident)
        expected = {
            (r["pid"], int(r["decoding"]["seed"]), c[0], str(c[1]))
            for r in eligible
            for c in all_conds
        }
        got = [_key(x) for x in rows]
        fails = []
        missing = sorted(expected - set(got))
        dup = sorted({k for k in got if got.count(k) > 1})
        extra = sorted(set(got) - expected)
        if missing:
            fails.append(f"missing conditions: {missing}")
        if dup:
            fails.append(f"duplicated conditions: {dup}")
        if extra:
            fails.append(f"unexpected conditions: {extra}")
        for s_ in rows:
            c = s_["checks"]
            if s_["press"] == "fullcache":
                if c["fullfull_max"] != 0.0:
                    fails.append(f"full/full logits differ: {c['fullfull_max']}")
                continue
            if c["n_nonfinite"]:
                fails.append(f"{s_['press']}: {c['n_nonfinite']} non-finite values")
            l0 = c.get("layer0_delta_max")
            if l0 is None or l0 > LAYER0_DELTA_TOL:
                fails.append(f"{s_['press']}: layer-0 Delta {l0} > {LAYER0_DELTA_TOL}")
            if s_["n_events"] < 1:
                fails.append(f"{s_['press']}: no eviction event")
        report = {
            "gate": "R.9 capture smoke",
            "passed": not fails,
            "failures": fails,
            "identity": ident,
            "expected_conditions": sorted(map(list, expected)),
            "rows": [
                {
                    k: s_.get(k)
                    for k in (
                        "pid",
                        "press",
                        "budget",
                        "n_events",
                        "checks",
                        "M_mean",
                        "Delta_mean",
                        "collateral_pooled",
                        "wallclock_s",
                        "full_branch_s",
                        "n_trace_tokens",
                    )
                }
                for s_ in rows
            ],
        }
        target = write_gate(base, SMOKE_PASS, SMOKE_FAIL, report)
        print(f"{'PASS' if not fails else 'FAIL'}: wrote {target}")
        return 0 if not fails else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

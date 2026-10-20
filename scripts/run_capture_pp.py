#!/usr/bin/env python3
"""Protected M / Delta / omega capture (Meeting 5, M5A-CAPTURE-PP; prepared
2026-10-20 -- HELD, not approved; no GPU run).

The same teacher-forced replay as ``scripts/run_capture.py`` (capture-b256),
on the PROTECTED presses (``kvcache.capture_protected``): for every reference
trace of the predeclared cohort the full branch is replayed once, then each
protected policy's compressed branch is replayed against it, producing M,
M_event, Delta, GQA collateral and omega with every live control of the
unprotected capture.

**Cohort.** The capture cohort is the 25 capture problems
(``subset.math500: 25``) restricted to the frozen strict-fit cohort
(``cohort_source``). Two of the 25 relax the B=256 budget under protection
(test/algebra/1349, test/precalculus/1199), so they are excluded, which
leaves 23 problems x the configured seeds. The protected accuracy compared
against these captures must be read on exactly the same keys.

**Smoke** (``smoke: true``) writes ``<base runs>/capture-pp-smoke-PASS.json``
only if every control holds:
- every expected condition is present exactly once;
- the full/full control is exactly zero;
- layer-0 Delta is within the algebra tolerance;
- there are no non-finite values and at least one eviction event per policy;
- every protected row read the prompt;
- with ``identity_control: true``, ``random_pp`` with the protected length
  forced to 0 reproduces the unprotected ``random`` capture of the same trace
  EXACTLY (M, M_event, Delta and the latent sums).

    python scripts/run_capture_pp.py \\
        --config configs/experiments/capture_pp_smoke_m5.yaml --dry-run
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
from kvcache.capture_protected import CAPTURE_PP_KIND, PROTECTED_POLICIES
from kvcache.cli import load_config, plan
from kvcache.datasets import load_benchmark
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    same_identity,
    write_gate,
)
from kvcache.runmeta import capture_runmeta

SCHEMA = "capture-pp-v1"
SMOKE_PASS = "capture-pp-smoke-PASS.json"
SMOKE_FAIL = "capture-pp-smoke-FAIL.json"
# fields that must be bitwise equal between random_pp@0 and random (identity)
IDENTITY_FIELDS = ("sum_by_layer_group", "n_measured_by_layer_group", "event_steps")
IDENTITY_LATENT = ("sum_omega", "n_finite", "sum_par", "sum_perp")


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
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else None
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def cohort_pids(cfg: dict[str, Any], subset_pids: list[str]) -> list[str]:
    """The predeclared capture cohort: the capture subset restricted to the
    frozen strict-fit cohort (no budget relaxation under protection)."""
    src = json.loads((PROJECT_ROOT / cfg["cohort_source"]).read_text())
    keep = set(src["pids"])
    return [p for p in subset_pids if p in keep]


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Every reason the config is not a protected-capture config."""
    bad = [p for p in cfg["presses"] if p not in PROTECTED_POLICIES]
    out = [f"not a protected capture policy: {p}" for p in bad]
    if "cohort_source" not in cfg:
        out.append("cohort_source (the strict-fit cohort) is required")
    if [str(b) for b in cfg["budgets"]] != ["256"]:
        out.append("the protected capture is registered at budget 256 only")
    return out


def _persisted(out_root: Path, ident: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = out_root / "math500"
    for path in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        for rec in _heal_jsonl(path):
            if not same_identity((rec.get("provenance") or {}), ident):
                raise SystemExit(
                    f"REFUSED: {path} holds a row from a different experiment "
                    "identity; start a new run namespace (old rows are kept)"
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


def smoke_gate_problem(base_runs: Path, ident: dict[str, Any]) -> str | None:
    """A campaign run needs a PASS smoke of the SAME implementation, model and
    basis (the smoke's config naturally differs). None when it holds."""
    path = base_runs / SMOKE_PASS
    if not path.exists():
        return f"no {SMOKE_PASS}"
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("passed") is not True:
        return f"{SMOKE_PASS} does not say passed"
    got = rec.get("identity") or {}
    for k in ("kind", "implementation_id", "model_revision", "basis_npz_sha256"):
        if got.get(k) != ident.get(k):
            return f"{SMOKE_PASS} is stale: {k} differs"
    return None


def identity_problems(base: dict[str, Any], pp: dict[str, Any]) -> list[str]:
    """Why ``random_pp`` with prompt length 0 does not reproduce ``random``."""
    out = [f"{f} differs" for f in IDENTITY_FIELDS if base.get(f) != pp.get(f)]
    lb, lp = base.get("latent") or {}, pp.get("latent") or {}
    out += [f"latent.{f} differs" for f in IDENTITY_LATENT if lb.get(f) != lp.get(f)]
    return out


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "capture_pp")
    problems = validate_config(cfg)
    if problems:
        raise SystemExit(f"REFUSED: {problems}")
    print(
        json.dumps(
            {
                "held": cfg.get("held", True),
                "presses": cfg["presses"],
                "budgets": cfg["budgets"],
                "cohort_source": cfg["cohort_source"],
                "smoke": bool(cfg.get("smoke")),
                "fullfull_control": bool(cfg.get("fullfull_control")),
                "identity_control": bool(cfg.get("identity_control")),
                "reference_limit": cfg.get("reference_limit"),
            }
        )
    )
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if cfg.get("held", True):
        raise SystemExit(
            "REFUSED: the protected capture is HELD (it needs its own smoke and "
            "campaign approval)"
        )
    if not args.i_understand:
        raise SystemExit("REFUSED: costly execution requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")

    import numpy as np
    import torch

    from kvcache.capture import record_full_branch, replay_condition
    from kvcache.capture_protected import (
        build_protected_capture_press,
        replay_condition_protected,
    )
    from kvcache.models import load_model_and_tokenizer
    from kvcache.presses import build_press, press_seed

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    out_root = Path(os.environ["REMOTE_RUNS_ROOT"])
    interval, tau = int(cfg["compression_interval"]), float(cfg["tau"])

    bdir = base / cfg["basis_run"] / "basis"
    header = json.loads((bdir / BASIS_HEADER).read_text(encoding="utf-8"))
    basis_sha = sha256_file(bdir / BASIS_NPZ)
    if basis_sha != header["npz_sha256"]:
        raise SystemExit("REFUSED: basis bytes do not match their frozen hash")
    if header.get("reference_run_name") != cfg["reference_run_name"]:
        raise SystemExit("REFUSED: basis was fitted on a different reference run")
    arr = np.load(bdir / BASIS_NPZ)

    ref_root = base / cfg["reference_run_name"]
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    benchmarks_cfg = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    problems_ = load_benchmark(
        "math500",
        benchmarks_cfg["math500"],
        subset=cfg["subset"]["math500"],
        allow_download=False,
    )
    pids = set(cohort_pids(cfg, [p.pid for p in problems_]))
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
    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][model_name]
    if header.get("model_revision") != str(model_cfg["revision"]):
        raise SystemExit("REFUSED: basis was fitted on a different model revision")
    ident = identity(
        CAPTURE_PP_KIND,
        cfg,
        model_revision=str(model_cfg["revision"]),
        reference_run=cfg["reference_run_name"],
        basis_npz_sha256=basis_sha,
    )
    if not cfg.get("smoke"):
        why = smoke_gate_problem(base, ident)
        if why:
            raise SystemExit(f"REFUSED: smoke gate: {why}")
    manifest = load_input_manifest()
    persisted = _persisted(out_root, ident)
    done = {_key(r) for r in persisted} if args.resume else set()
    if not args.resume and persisted:
        raise SystemExit("REFUSED: run namespace not empty; pass --resume")
    model, _tok = load_model_and_tokenizer(
        model_cfg, allow_download=False, attn_implementation="eager"
    )
    basis = {
        k: torch.tensor(arr[k], device=model.device, dtype=torch.float32)
        for k in ("scale", "basis")
    }
    conds = [(p, str(b)) for p in cfg["presses"] for b in cfg["budgets"]]
    if cfg.get("fullfull_control"):
        conds = [("fullcache", "full"), *conds]
    identity_report: list[dict[str, Any]] = []
    for r in eligible:
        seed = int(r["decoding"]["seed"])
        todo = [c for c in conds if (r["pid"], seed, c[0], c[1]) not in done]
        if not todo and not cfg.get("identity_control"):
            continue
        rec = tokens[_reference_key(r)]
        prompt, trace = rec["prompt_token_ids"], rec["trace_token_ids"]
        check_inputs(manifest, "capture", r["pid"], seed, inputs_sha256(prompt, trace))
        t0 = time.monotonic()
        full = record_full_branch(model, prompt, trace, interval=interval)
        full_s = time.monotonic() - t0
        runmeta = capture_runmeta(cfg["name"], args.config, seed)
        common = {
            "interval": interval,
            "tau": tau,
            "trigger_step": int(cfg.get("trigger_step", 130)),
        }
        for press_name, budget in todo:
            t1 = time.monotonic()
            if press_name == "fullcache":
                out = replay_condition(
                    model, None, full, prompt, trace, budget=None, **common
                )
            else:
                press = build_protected_capture_press(
                    press_name,
                    budget=int(budget),
                    compression_interval=interval,
                    seed=press_seed(r["pid"], seed),
                )
                out = replay_condition_protected(
                    model,
                    press,
                    full,
                    prompt,
                    trace,
                    budget=int(budget),
                    basis=basis,
                    **common,
                )
            sampled = out.pop("latent_sampled", None)
            if sampled is not None:
                pid_hash = hashlib.sha256(r["pid"].encode()).hexdigest()[:16]
                npz = (
                    out_root
                    / "latent"
                    / f"{press_name}-{budget}-s{seed}"
                    / (f"{pid_hash}.npz")
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
            print(
                f"{r['pid']} s{seed} {press_name}-{budget}: events={row['n_events']} "
                f"M={row['M_mean']} Delta={row['Delta_mean']} "
                f"omega={(row.get('latent') or {}).get('omega_trace')} "
                f"{row['wallclock_s']:.0f}s",
                flush=True,
            )
        if cfg.get("smoke") and cfg.get("identity_control"):
            # random_pp with the protected length forced to 0 must reproduce
            # the unprotected random capture of this trace exactly
            kw = {**common, "budget": 256, "basis": basis}
            seed_p = press_seed(r["pid"], seed)
            ref = replay_condition(
                model,
                build_press(
                    "random",
                    budget=256,
                    compression_interval=interval,
                    seed=seed_p,
                    instrumented=True,
                ),
                full,
                prompt,
                trace,
                **kw,
            )
            pp = replay_condition_protected(
                model,
                build_protected_capture_press(
                    "random_pp",
                    budget=256,
                    compression_interval=interval,
                    seed=seed_p,
                    forced_prompt_len=0,
                ),
                full,
                prompt,
                trace,
                **kw,
            )
            identity_report.append(
                {
                    "pid": r["pid"],
                    "seed": seed,
                    "problems": identity_problems(_round(ref), _round(pp)),
                }
            )
        del full
        torch.cuda.empty_cache()

    if cfg.get("smoke"):
        rows = _persisted(out_root, ident)
        expected = {
            (r["pid"], int(r["decoding"]["seed"]), c[0], c[1])
            for r in eligible
            for c in conds
        }
        got = [_key(x) for x in rows]
        fails = []
        if sorted(expected - set(got)):
            fails.append(f"missing conditions: {sorted(expected - set(got))}")
        dup = sorted({k for k in got if got.count(k) > 1})
        if dup:
            fails.append(f"duplicated conditions: {dup}")
        if sorted(set(got) - expected):
            fails.append(f"unexpected conditions: {sorted(set(got) - expected)}")
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
            prot = s_.get("protect") or {}
            if prot.get("prompt_len") != s_["prompt_tokens"]:
                fails.append(
                    f"{s_['press']}: press read prompt length {prot.get('prompt_len')}"
                )
        if cfg.get("identity_control"):
            if not identity_report:
                fails.append("identity control did not run")
            for item in identity_report:
                if item["problems"]:
                    fails.append(
                        f"identity control {item['pid']} s{item['seed']}: "
                        f"{item['problems']}"
                    )
        report = {
            "gate": "M5A-CAPTURE-PP smoke",
            "passed": not fails,
            "failures": fails,
            "identity": ident,
            "expected_conditions": sorted(map(list, expected)),
            "identity_control": identity_report,
            "rows": [
                {
                    k: s_.get(k)
                    for k in (
                        "pid",
                        "seed",
                        "press",
                        "budget",
                        "n_events",
                        "checks",
                        "protect",
                        "M_mean",
                        "Delta_mean",
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

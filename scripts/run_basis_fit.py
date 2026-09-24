#!/usr/bin/env python3
"""Amendment R.6 / Q.6 fit: the frozen full-only PCA basis, per layer, k = 64.

Full-cache reference traces of the Q.2 subset only (the compressed condition is
never seen: circularity is a stop condition). For each trace, up to 256
uniformly spaced decode positions fixed from the trace length alone
(``kvcache.capture.sampled_decode_positions``), the output of every decoder
block at the prediction position (before the final norm). Problems weigh
equally, seeds equally within a problem, positions equally within a trace.
float64 moments on CPU; coordinate-wise standardisation (the existing
``whiten()``, not covariance whitening); exact per-layer eigendecomposition.

The teacher-forced states are taken from one forward over prompt + trace (a
full cache is never pruned, so this is the same conditioning as stepwise
decoding; it differs only at kernel rounding, which a basis fit does not see).

Writes ``<run>/basis/q6_basis.npz`` then, LAST, ``q6_basis.json`` holding its
sha256, the input hashes and diagnostics -- the header is the queue marker, so
a crash can never leave a basis that looks finished. The capture refuses a
basis whose bytes do not match the header hash.

    python scripts/run_basis_fit.py \
        --config configs/experiments/basis_fit_q6_r9.yaml --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_replay import _load_jsonl, _load_token_records, _reference_key

from kvcache.capture import (
    MomentAccumulator,
    sampled_decode_positions,
    validate_basis_arrays,
)
from kvcache.cli import load_config, plan
from kvcache.provenance import (
    check_inputs,
    identity,
    inputs_sha256,
    load_input_manifest,
    same_identity,
    write_json_atomic,
)
from kvcache.datasets import load_benchmark
from kvcache.runmeta import capture_runmeta

BASIS_NPZ = "q6_basis.npz"
BASIS_HEADER = "q6_basis.json"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    p.add_argument("--resume", action="store_true", help="ignored (single pass)")
    p.add_argument("--seeds", default="", help="ignored: the fit needs every seed")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    plan(cfg, args.config, "basisfit")
    k, n_pos = int(cfg["k"]), int(cfg["n_positions"])
    print(
        json.dumps(
            {"k": k, "n_positions": n_pos, "reference": cfg["reference_run_name"]}
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

    from kvcache.models import load_model_and_tokenizer

    base = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    out_dir = Path(os.environ["REMOTE_RUNS_ROOT"]) / "basis"
    models_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"]
    model_cfg = models_cfg[cfg["models"][0]]
    ident = identity(
        "basis",
        cfg,
        model_revision=str(model_cfg["revision"]),
        reference_run=cfg["reference_run_name"],
    )
    if (out_dir / BASIS_HEADER).exists():
        # A header alone is not a finished fit: the bytes it names must exist
        # and hash to what it recorded, and it must be THIS experiment's fit.
        head = json.loads((out_dir / BASIS_HEADER).read_text(encoding="utf-8"))
        npz_path = out_dir / head.get("npz", BASIS_NPZ)
        if not npz_path.exists() or sha256_file(npz_path) != head.get("npz_sha256"):
            raise SystemExit(
                f"REFUSED: {out_dir / BASIS_HEADER} names a missing or altered "
                "npz; start a new run namespace (nothing is overwritten)"
            )
        if not same_identity(head.get("identity"), ident):
            raise SystemExit(
                "REFUSED: the frozen basis belongs to a different experiment "
                "identity; start a new run namespace"
            )
        print(f"basis already frozen and verified: {out_dir / BASIS_HEADER}")
        return 0
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
    seeds = {int(s) for s in cfg["seeds"]}
    model_name = cfg["models"][0]
    eligible = sorted(
        (
            r
            for r in rows
            if r["pid"] in pids
            and int(r["decoding"]["seed"]) in seeds
            and r["model"] == model_name
        ),
        key=lambda r: (r["pid"], int(r["decoding"]["seed"])),
    )
    per_pid: dict[str, int] = defaultdict(int)
    for r in eligible:
        per_pid[r["pid"]] += 1
    expected = len(pids) * len(seeds)
    if len(eligible) != expected:
        raise SystemExit(
            f"fit needs {expected} reference traces, found {len(eligible)}"
        )

    model, _tok = load_model_and_tokenizer(model_cfg, allow_download=False)
    layers = model.model.layers
    n_layers, dim = len(layers), int(model.config.hidden_size)
    manifest = load_input_manifest()
    acc = MomentAccumulator(n_layers, dim)
    input_hash = hashlib.sha256()
    started = time.monotonic()
    grab: dict[str, Any] = {}

    def hook(i: int):
        def fn(_m: Any, _a: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            grab[i] = h[0, grab["idx"]].to("cpu", torch.float64)

        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        for n, r in enumerate(eligible):
            rec = tokens[_reference_key(r)]
            prompt, trace = rec["prompt_token_ids"], rec["trace_token_ids"]
            check_inputs(
                manifest,
                "basis",
                r["pid"],
                int(r["decoding"]["seed"]),
                inputs_sha256(prompt, trace),
            )
            pos = sampled_decode_positions(len(trace), n_pos)
            if not pos:
                raise SystemExit(f"trace too short to sample: {r['pid']}")
            input_hash.update(
                json.dumps([list(_reference_key(r)), prompt, trace]).encode()
            )
            ids = list(prompt) + list(trace[: len(trace) - 1])
            grab["idx"] = torch.tensor(
                [len(prompt) + j - 1 for j in pos], device=model.device
            )
            with torch.inference_mode():
                # The decoder stack only: the LM head over ~16k positions would
                # materialise ~5 GB of logits the fit never reads.
                model.model(
                    input_ids=torch.tensor([ids], device=model.device), use_cache=False
                )
            x = np.stack([grab[i].numpy() for i in range(n_layers)])
            weight = 1.0 / (len(pids) * per_pid[r["pid"]] * len(pos))
            acc.add(x, weight)
            print(
                f"[{n + 1}/{len(eligible)}] {r['pid']} s{r['decoding']['seed']} "
                f"T={len(trace)} n_pos={len(pos)} {time.monotonic() - started:.0f}s",
                flush=True,
            )
    finally:
        for h in handles:
            h.remove()

    fit = acc.solve(k)
    mean = fit["mean"].astype(np.float32)
    scale = fit["scale"].astype(np.float32)
    basis = fit["basis"].astype(np.float32)
    info = validate_basis_arrays(mean, scale, basis)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / BASIS_NPZ
    tmp_npz = out_dir / f".{BASIS_NPZ}.tmp-{os.getpid()}.npz"
    np.savez(
        tmp_npz,
        mean=mean,
        scale=scale,
        basis=basis,
        eigenvalues=fit["eigenvalues"],
        explained_fraction=fit["explained_fraction"],
    )
    os.replace(tmp_npz, npz)
    runmeta = capture_runmeta(cfg["name"], args.config, 0)
    header = {
        "schema_version": "q6-basis-v1",
        "passed": True,
        "status": "FROZEN",
        "identity": ident,
        "amendment": "R.6 / R.9",
        "npz": BASIS_NPZ,
        "npz_sha256": sha256_file(npz),
        "input_sha256": input_hash.hexdigest(),
        "reference_run_name": cfg["reference_run_name"],
        "n_traces": len(eligible),
        "n_problems": len(pids),
        "k": k,
        "n_positions_max": n_pos,
        "representation": "decoder block output at the prediction position "
        "(pre final norm), float64 moments, coordinate-wise standardisation",
        "weights": "problems equal; seeds equal within problem; positions equal "
        "within trace",
        "max_orthonormality_error": info["max_orthonormality_error"],
        "explained_fraction_by_layer": [float(x) for x in fit["explained_fraction"]],
        "top_eigenvalue_by_layer": [float(x) for x in fit["eigenvalues"][:, 0]],
        "model_revision": str(model_cfg["revision"]),
        "source_tree_hash": runmeta.get("source_tree_hash", ""),
        "commit": runmeta.get("commit", ""),
        "wallclock_s": time.monotonic() - started,
    }
    write_json_atomic(out_dir / BASIS_HEADER, header)
    print(json.dumps({k_: header[k_] for k_ in ("npz_sha256", "n_traces")}))
    print(f"FROZEN: {out_dir / BASIS_HEADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

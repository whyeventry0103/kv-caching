#!/usr/bin/env python3
"""Protected R-KV (rkv_pp_v1) real-model parity on CPU -- Meeting 5, M5-A.

The unit tests (tests/test_rkv_prompt_protect.py) replace the R-KV scorer with a
fake in both arms, so they isolate the wrapper. This check runs the REAL scorer on
REAL model tensors, without a GPU, on the small pinned model whose fixed bank
already exists (fc-q15b):

1. identity control -- base ``rkv`` and ``rkv_pp_v1`` with ``forced_prompt_len=0``
   are replayed over the same fixed trace. Every prune must return byte-identical
   keys and values (hashed per layer and prune) and the TV profiles must be
   identical. Protection off has to BE the parity-certified base path.
2. replay -- ``rkv_pp_v1`` with the real prompt. The press's own live check
   raises if any prompt position or any of the 8 most recent positions is evicted,
   or a position is duplicated, in any layer or KV head; target = max(B, P + 64)
   and every post-prune size equals it; every layer pruned; pre-eviction TV 0.

Pass/fail is ``rkv_prompt_protect.smoke_failures`` -- the same checks the GPU
smoke (scripts/rkv_pp_smoke.py) applies, minus free generation.

It is a CPU engineering check, not a scientific row: nothing is appended to any
run, and passing it does not register or promote rkv_pp_v1 (that needs the GPU
smoke and its own approval). Model weights load from the local cache only, at the
pinned revision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_replay import _load_jsonl, _load_token_records, _reference_key

from kvcache.presses.prompt_protect import (
    effective_budget,
    effective_prunes,
    select_smoke_traces,
)
from kvcache.presses.rkv_prompt_protect import (
    first_mismatch,
    record_prune_outputs,
    smoke_failures,
)

DEFAULT_MODEL = "r1_distill_qwen_1_5b"
DEFAULT_REFERENCE = "fc-q15b"


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--reference-run", default=DEFAULT_REFERENCE)
    p.add_argument("--seed", type=int, default=11)  # the bank's first seed
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--compression-interval", type=int, default=64)
    p.add_argument("--replay-tokens", type=int, default=320)
    p.add_argument("--min-trace-tokens", type=int, default=320)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--i-understand", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    plan = {
        "model": args.model,
        "reference_run": args.reference_run,
        "budget": args.budget,
        "compression_interval": args.compression_interval,
        "replay_tokens": args.replay_tokens,
        "device": "cpu",
        "threads": args.threads,
    }
    print(json.dumps(plan, sort_keys=True))
    if args.dry_run:
        print("DRY RUN: planned only; no model, dataset, or tensor work")
        return 0
    if not args.i_understand:
        raise SystemExit("REFUSED: loading model weights requires --i-understand")
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: model execution requires KVCACHE_REMOTE=1")

    import torch
    from transformers import AutoModelForCausalLM

    import kvpress
    from kvcache.models import _revision
    from kvcache.presses import build_press
    from kvcache.presses.rkv_prompt_protect import (
        FAMILY,
        build_rkv_prompt_protected_press,
    )
    from kvcache.replay import TransformersReplayBackend, paired_teacher_forced_replay

    torch.set_num_threads(int(args.threads))
    base_root = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    ref_root = base_root / args.reference_run
    rows = _load_jsonl(ref_root)
    tokens = _load_token_records(ref_root)
    candidates = [
        {
            "pid": r["pid"],
            "prompt_len": len(tokens[_reference_key(r)]["prompt_token_ids"]),
            "trace_len": len(tokens[_reference_key(r)]["trace_token_ids"]),
            "key": _reference_key(r),
        }
        for r in rows
        if r["model"] == args.model and int(r["decoding"]["seed"]) == args.seed
    ]
    chosen = select_smoke_traces(candidates, int(args.min_trace_tokens))
    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][args.model]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_cfg["handle"]),
        revision=_revision(model_cfg),
        local_files_only=True,
        dtype=getattr(torch, str(model_cfg.get("dtype", "bfloat16"))),
        device_map={"": "cpu"},
    )
    model.eval()

    def replay(press: Any, prompt_ids: list[int], trace_ids: list[int]) -> Any:
        backend = TransformersReplayBackend(model)
        with torch.inference_mode(), press(model):
            return paired_teacher_forced_replay(
                prompt_ids,
                trace_ids,
                cache_factory=backend.cache_factory,
                forward=backend.forward,
                press=press,
            )

    kw = dict(
        budget=args.budget,
        compression_interval=args.compression_interval,
        kvpress_module=kvpress,
        instrumented=True,
    )
    records: list[dict[str, Any]] = []
    for c in chosen:
        rec = tokens[c["key"]]
        prompt_ids = list(rec["prompt_token_ids"])
        trace_ids = list(rec["trace_token_ids"])[: int(args.replay_tokens)]
        out: dict[str, Any] = {"kind": "identity", "pid": c["pid"]}
        t0 = time.monotonic()
        try:
            base = build_press("rkv", **kw)
            prot = build_rkv_prompt_protected_press(FAMILY, forced_prompt_len=0, **kw)
            log_b, log_p = record_prune_outputs(base), record_prune_outputs(prot)
            prof_b = replay(base, prompt_ids, trace_ids)
            prof_p = replay(prot, prompt_ids, trace_ids)
            out.update(
                prompt_len=len(prompt_ids),
                n_replayed=len(trace_ids),
                n_prunes_base=len(log_b),
                n_prunes_protected_off=len(log_p),
                first_kv_mismatch=first_mismatch(log_b, log_p),
                tv_identical=prof_b.estimate == prof_p.estimate,
                mean_tv=sum(prof_b.estimate) / len(prof_b.estimate),
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        out["wallclock_s"] = round(time.monotonic() - t0, 1)
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

        out = {
            "kind": "replay",
            "pid": c["pid"],
            "prompt_len": len(prompt_ids),
            "n_replayed": len(trace_ids),
            "n_layers": len(model.model.layers),
        }
        t0 = time.monotonic()
        try:
            prot = build_rkv_prompt_protected_press(FAMILY, **kw)
            prof = replay(prot, prompt_ids, trace_ids)
            pruned = effective_prunes(prot.compressions)
            first = min((int(e["step"]) for e in pruned), default=None)
            out.update(
                protect=prot.protect.report(),
                expected_target=effective_budget(
                    args.budget, len(prompt_ids), prot.protect.min_trace_slots
                )[0],
                n_effective_prunes=len(pruned),
                first_effective_prune=first,
                pre_eviction_max_tv=max(prof.estimate[: (first or 0) + 1], default=0.0),
                post_prune_sizes=sorted({int(e["after"]) for e in pruned}),
                mean_tv=sum(prof.estimate) / len(prof.estimate),
            )
        except RuntimeError as exc:
            out["raised"] = str(exc)
        out["wallclock_s"] = round(time.monotonic() - t0, 1)
        records.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    fails = smoke_failures(records)  # the GPU smoke's own checks
    report = {
        "check": f"{FAMILY} real-model CPU parity (engineering check, not a row)",
        "passed": not fails,
        "status": "PASS" if not fails else "FAIL",
        "failures": fails,
        "plan": plan,
        "selected": [
            {k: c[k] for k in ("pid", "prompt_len", "trace_len")} for c in chosen
        ],
        "versions": {
            "torch": torch.__version__,
            "kvpress": getattr(kvpress, "__version__", None),
        },
        "finished_at": datetime.now(UTC).isoformat(),
        "records": records,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "failures")}))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

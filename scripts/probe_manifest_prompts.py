#!/usr/bin/env python3
"""Prompt lengths of the frozen replication splits and their fit at each
candidate budget (M5-B design input; prepared 2026-11-13). CPU only: loads the
pinned tokenizer and the pinned datasets OFFLINE, never model weights; runs on
the GPU host (where the pins are cached) without touching a GPU.

Prompt length is a property of the inputs, so this is read before any
compressed outcome. It answers plan §8.2's first question for each dataset:
how many problems fit ``P + 64 <= B`` (the strict-fit cohort) and how much
trace space remains under ``B_eff = max(B, P + 64)``
(``kvcache.manifest_capacity``).

    python scripts/probe_manifest_prompts.py --dry-run
    KVCACHE_REMOTE=1 python scripts/probe_manifest_prompts.py \\
        --out results/diag/meeting5/replication/prompt_fit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.datasets import load_benchmark
from kvcache.manifest_capacity import prompt_fit
from kvcache.manifest_problems import (
    frozen_split_pids,
    load_frozen_pool,
    manifest_key,
    select_frozen_split,
)

MANIFEST = PROJECT_ROOT / "configs/meeting5/problem_manifests.json"


def _yaml(path: Path) -> Any:
    import yaml

    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benches", default="gsm8k,bbh_lde5")
    p.add_argument("--splits", default="calibration,confirmation")
    p.add_argument("--budgets", default="256,1024")
    p.add_argument("--model", default="r1_distill_qwen_7b")
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    return p


def resolve(benches: list[str], splits: list[str]) -> dict[str, dict[str, Any]]:
    """Each (bench, split)'s frozen IDs from the manifest alone."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cfgs = _yaml(PROJECT_ROOT / "configs/benchmarks.yaml")["benchmarks"]
    out: dict[str, dict[str, Any]] = {}
    for bench in benches:
        key = manifest_key(bench, cfgs[bench])
        out[bench] = {
            "cfg": cfgs[bench],
            "n_pool": int(manifest[key]["n_pool"]),
            "splits": {
                s: frozen_split_pids(
                    manifest, key, s, pinned_revision=str(cfgs[bench]["revision"])
                )
                for s in splits
            },
        }
    return out


def main() -> int:
    args = _parser().parse_args()
    benches = [b for b in args.benches.split(",") if b]
    splits = [s for s in args.splits.split(",") if s]
    budgets = [int(b) for b in args.budgets.split(",") if b]
    plan = resolve(benches, splits)
    print(
        json.dumps(
            {b: {s: len(v) for s, v in d["splits"].items()} for b, d in plan.items()}
        )
    )
    if args.dry_run:
        print("DRY RUN: manifest resolved; no tokenizer, dataset or model loaded")
        return 0
    from kvcache.models import load_tokenizer, render_prompt, tokenize_rendered_prompt

    model_cfg = _yaml(PROJECT_ROOT / "configs/models.yaml")["models"][args.model]
    tok = load_tokenizer(model_cfg, allow_download=False)
    report: dict[str, Any] = {
        "model": args.model,
        "model_revision": str(model_cfg["revision"]),
        "rule": "B_eff = max(B, P + 64) (kvcache.presses.prompt_protect)",
        "benches": {},
    }
    for bench, d in plan.items():
        pool = load_frozen_pool(bench, d["cfg"], hf_loader=load_benchmark)
        entry: dict[str, Any] = {"revision": str(d["cfg"]["revision"])}
        for split, pids in d["splits"].items():
            problems = select_frozen_split(pool, pids, n_pool=d["n_pool"])
            lens = {
                p.pid: len(
                    tokenize_rendered_prompt(
                        tok, render_prompt(tok, p.question, model_cfg)
                    )["input_ids"]
                )
                for p in problems
            }
            entry[split] = {
                "prompt_lens": lens,
                "fit": {str(b): prompt_fit(lens, b) for b in budgets},
            }
        report["benches"][bench] = entry
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    for bench, entry in report["benches"].items():
        for split in splits:
            for b, f in entry[split]["fit"].items():
                print(
                    f"{bench}/{split} B={b}: strict-fit {f['n_strict_fit']}/{f['n']}"
                    f"; P median {_median(entry[split])}"
                    f"; trace slots (strict) median "
                    f"{f['trace_slots_strict_fit'].get('median')}"
                )
    return 0


def _median(split_entry: dict[str, Any]) -> float:
    import statistics

    return float(statistics.median(split_entry["prompt_lens"].values()))


if __name__ == "__main__":
    raise SystemExit(main())

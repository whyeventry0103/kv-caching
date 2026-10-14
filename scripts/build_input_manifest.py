#!/usr/bin/env python3
"""Freeze a Meeting #3 input manifest: the reference inputs every row must carry.

For one model's full-cache reference run and the frozen Q.2 subset, records
``inputs_sha256(prompt)`` per problem and ``inputs_sha256(prompt, trace)`` per
(problem, seed) -- exactly what ``kvcache.provenance.check_inputs`` compares
against. Proven by rebuilding the committed Qwen manifest byte-for-byte before
it is used for another model (R.17.3).

    python scripts/build_input_manifest.py --runs-root <runs> \\
        --reference-run fc-llama50 --model r1_distill_llama_8b \\
        --out configs/meeting3/input_manifest_llama8b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kvcache.provenance import inputs_sha256

SEEDS = (11, 23, 47, 89)


DEFAULT_PIDS = "configs/meeting3/q2_subset_pids.json"


def build(
    runs_root: Path, reference_run: str, model: str, pids_file: str = DEFAULT_PIDS
) -> dict:
    from run_replay import _load_jsonl, _load_token_records, _reference_key

    pids = json.loads((PROJECT_ROOT / pids_file).read_text())["pids"]
    ref = runs_root / reference_run
    rows = [r for r in _load_jsonl(ref) if r["model"] == model]
    tokens = _load_token_records(ref)
    by = {(r["pid"], int(r["decoding"]["seed"])): r for r in rows}
    prompts, traces = {}, {}
    for pid in pids:
        for seed in SEEDS:
            rec = tokens[_reference_key(by[(pid, seed)])]
            traces[f"{pid}|{seed}"] = inputs_sha256(
                rec["prompt_token_ids"], rec["trace_token_ids"]
            )
            p = inputs_sha256(rec["prompt_token_ids"])
            if prompts.setdefault(pid, p) != p:
                raise SystemExit(f"prompt tokens differ across seeds for {pid}")
    return {
        "hash": "kvcache.provenance.inputs_sha256",
        "model": model,
        "prompts": prompts,
        "reference_run": reference_run,
        "schema": "input-manifest-v1",
        "traces_prompt_plus_trace": traces,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--reference-run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    # Meeting 5 (2026-10-14): the 100-problem discovery cohort needs prompt
    # hashes for all 100 problems; the default list is unchanged.
    ap.add_argument("--pids", default=DEFAULT_PIDS, help="frozen pid-list JSON")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "input-manifest", "out": str(a.out)}))
        return 0
    m = build(a.runs_root, a.reference_run, a.model, a.pids)
    # no trailing newline: byte-identical to the committed Qwen manifest
    a.out.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
    print(
        f"wrote {a.out}: {len(m['prompts'])} prompts, "
        f"{len(m['traces_prompt_plus_trace'])} traces"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixtures for smoke-gate tests (review of 6b409001, R2): a small frozen
reference bank under a runs root, and a fully MEASURED PASS marker for either
smoke family, consistent with that bank -- the positive control the runtime
gates must accept, which each test then corrupts one way."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def reference_bank(runs_root: Path, smoke_cfg: dict[str, Any], n: int = 6) -> None:
    """Rows + token sidecars for the first ``n`` frozen problems at the config's
    model and first seed; prompt lengths 10, 17, 24, ... (all fit B=256)."""
    from kvcache.read_validation import frozen_pids

    pids = frozen_pids(int(smoke_cfg["subset"]["math500"]), root=ROOT)[:n]
    model, seed = smoke_cfg["models"][0], int(smoke_cfg["seeds"][0])
    ref = runs_root / smoke_cfg["reference_run_name"]
    rows, toks = [], []
    for i, pid in enumerate(pids):
        rows.append(
            {
                "pid": pid,
                "model": model,
                "bench": "math500",
                "press": "fullcache",
                "budget": "full",
                "decoding": {"seed": seed},
            }
        )
        toks.append(
            {
                "pid": pid,
                "model": model,
                "bench": "math500",
                "press": "fullcache",
                "budget": "full",
                "seed": seed,
                "prompt_token_ids": list(range(1000, 1010 + 7 * i)),
                "trace_token_ids": list(
                    range(int(smoke_cfg["min_trace_tokens"]) + 5 + i)
                ),
            }
        )
    (ref / "math500").mkdir(parents=True, exist_ok=True)
    (ref / "token_ids").mkdir(parents=True, exist_ok=True)
    (ref / "math500" / f"fullcache-full-s{seed}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (ref / "token_ids" / f"fullcache-full-s{seed}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in toks) + "\n"
    )


def _replay(trace: dict[str, Any], press: str, target: int) -> dict[str, Any]:
    return {
        "kind": "replay",
        "pid": trace["pid"],
        "press": press,
        "prompt_len": trace["prompt_len"],
        "n_replayed": 1536,
        "n_layers": 28,
        "inputs_sha256": trace["inputs_sha256"],
        "protect": {
            "prompt_len": trace["prompt_len"],
            "target": target,
            "n_layers_pruned": 28,
            "n_prunes": 84,
        },
        "expected_target": target,
        "first_effective_prune": 128,
        "pre_eviction_max_tv": 0.0,
        "post_prune_sizes": [target],
        "mean_tv": 0.05,
    }


def _generation(trace: dict[str, Any], press: str) -> dict[str, Any]:
    return {
        "kind": "generation",
        "pid": trace["pid"],
        "press": press,
        "prompt_len": trace["prompt_len"],
        "n_tokens": 700,
        "protect": {"prompt_len": trace["prompt_len"], "n_prunes": 3},
    }


def measured_marker(
    runs_root: Path, smoke_cfg: dict[str, Any], family: str
) -> dict[str, Any]:
    """A valid PASS for ``family`` ("protsel" or "rkvpp"), written and returned."""
    from kvcache import smoke_validation as sv

    traces = sv.reference_traces(runs_root, smoke_cfg, root=ROOT)
    target = int(smoke_cfg["budgets"][0])
    if family == "protsel":
        from kvcache.presses import protected_selectors as mod

        presses = sorted(str(p).lower() for p in smoke_cfg["presses"])
        ident_press = str(smoke_cfg["identity_family"])
        ident = {"identity_prunes": 644, "identity_mismatched_heads": 0}
        extra = {"families_tested": presses}
    else:
        from kvcache.presses import rkv_prompt_protect as mod

        presses = [mod.FAMILY]
        ident_press = mod.FAMILY
        ident = {
            "n_prunes_base": 112,
            "n_prunes_protected_off": 112,
            "first_kv_mismatch": None,
            "tv_identical": True,
        }
        extra = {}
    records = []
    for t in traces:
        for p in presses:
            records += [_replay(t, p, target), _generation(t, p)]
        records.append(
            {"kind": "identity", "pid": t["pid"], "press": ident_press, **ident}
        )
    marker = {
        "passed": True,
        "status": "PASS",
        "failures": [],
        "identity": mod.smoke_identity(smoke_cfg),
        "selected": traces,
        "records": records,
        **extra,
    }
    (runs_root / mod.SMOKE_MARKER).write_text(json.dumps(marker))
    return marker


def write(runs_root: Path, marker_name: str, marker: dict[str, Any]) -> None:
    (runs_root / marker_name).write_text(json.dumps(marker))

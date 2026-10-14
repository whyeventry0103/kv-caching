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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-tokenizer", action="store_true")
    ap.add_argument("--i-understand", action="store_true")
    a = ap.parse_args()
    cfg = _yaml(PROJECT_ROOT / a.config)
    print(json.dumps({"config": a.config, "held": cfg.get("held"), **counts(cfg)}))
    if a.dry_run:
        d = design(
            cfg, [f"b{cfg['design_seed']}-{i:04d}" for i in range(cfg["n_blocks"])]
        )
        print(json.dumps(d, indent=1, sort_keys=True))
        print("DRY RUN: design and counts only; no tokenizer, model or GPU")
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
    raise SystemExit(
        "REFUSED: the GPU execution path is wired in kvcache.information_probe "
        "(TransformersBackend, KVPressHandle) but is enabled only with its "
        "approval and campaign cap"
    )


if __name__ == "__main__":
    raise SystemExit(main())

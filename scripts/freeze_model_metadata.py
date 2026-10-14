#!/usr/bin/env python3
"""Freeze architecture metadata of pinned models (review of 946c76a, issue 1).

Smoke gates must check layer coverage against the MODEL, not against a number
the smoke record supplies itself. This reads each model's ``config.json`` at its
pinned revision from the local Hugging Face cache (no weights, no download) and
writes layer/head counts plus the file's sha256 to
``configs/meeting5/model_metadata.json``; ``kvcache.smoke_validation`` refuses
when the recorded revision differs from configs/models.yaml.

    python scripts/freeze_model_metadata.py --dry-run
    KVCACHE_REMOTE=1 python scripts/freeze_model_metadata.py --models r1_distill_qwen_7b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = "configs/meeting5/model_metadata.json"
FIELDS = ("num_hidden_layers", "num_attention_heads", "num_key_value_heads")


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["r1_distill_qwen_7b"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps({"models": a.models, "out": OUT}))
    if a.dry_run:
        print("DRY RUN: nothing read")
        return 0
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: the model cache is remote-only")
    from huggingface_hub import hf_hub_download

    models = yaml.safe_load((PROJECT_ROOT / "configs/models.yaml").read_text())
    out = {}
    for name in a.models:
        m = models["models"][name]
        p = Path(
            hf_hub_download(
                m["handle"],
                "config.json",
                revision=m["revision"],
                local_files_only=True,
            )
        )
        cfg = json.loads(p.read_text())
        out[name] = {
            "handle": m["handle"],
            "revision": m["revision"],
            "config_json_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            **{f: int(cfg[f]) for f in FIELDS},
        }
    path = PROJECT_ROOT / OUT
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

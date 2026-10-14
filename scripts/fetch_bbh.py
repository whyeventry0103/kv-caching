#!/usr/bin/env python3
"""Fetch the owner-pinned BBH logical_deduction_five_objects file (Meeting 5
M5-B; dataset PREPARATION only, pinned 2026-10-14).

Downloads exactly ``kvcache.bbh.raw_url()`` (the pinned commit), refuses it
unless its git blob hash equals the pinned blob and it parses to the expected
250 single-option examples, and only then writes it atomically under
``$REMOTE_DATA_ROOT/bbh/<commit>/``. Requires --allow-dataset-download.

    python scripts/fetch_bbh.py --dry-run
    python scripts/fetch_bbh.py --allow-dataset-download
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache import bbh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-dataset-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(json.dumps({"url": bbh.raw_url(), "blob": bbh.PINNED_BLOB}))
    if a.dry_run:
        print("DRY RUN: nothing downloaded")
        return 0
    if not a.allow_dataset_download:
        raise SystemExit("REFUSED: downloading requires --allow-dataset-download")
    from urllib.request import urlopen

    target = bbh.data_path(Path(os.environ["REMOTE_DATA_ROOT"]))
    with urlopen(bbh.raw_url(), timeout=60) as resp:  # noqa: S310 -- pinned URL
        data = resp.read()
    problems = bbh.parse_bbh(data, expected_blob=bbh.PINNED_BLOB)
    if len(problems) != bbh.EXPECTED_N:
        raise SystemExit(
            f"REFUSED: {len(problems)} examples, expected {bbh.EXPECTED_N}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)
    assert len(bbh.load_bbh(target)) == bbh.EXPECTED_N
    labels = sorted({p.gold_answer for p in problems})
    print(json.dumps({"wrote": str(target), "n": len(problems), "labels": labels}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate a committed smoke-evidence bundle without a GPU (Meeting 5 review, G0.1).

Checks, against the committed specification at the bundle's producing commit:

1. every bundled file hashes to the manifest value;
2. the PASS record says ``passed: true``, ``status: "PASS"``, lists no failures,
   and ``kvcache.presses.protected_selectors.smoke_failures`` finds none when
   re-applied to its records;
3. its identity equals the identity recomputed from a clean export of the
   producing commit (generation code, smoke config, model revision, input
   manifest, reference run) -- so the check stays valid after later code changes;
4. the executed record counts match the manifest's ``design_executed``;
5. the timing block is internally consistent (end after start, hours match).

Exit 0 when every check holds, 2 otherwise. Prints one JSON verdict.

    python scripts/validate_smoke_bundle.py --bundle results/diag/paper_audit/wp4_smoke
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvcache.presses.protected_selectors import smoke_failures

MARKER = "protected-selectors-smoke-PASS.json"


def _identity_at(commit: str, config: str) -> dict:
    """smoke_identity computed by the producing commit's own code and files."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", commit],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        code = (
            "import json, sys, yaml\n"
            "from pathlib import Path\n"
            "from kvcache.presses.protected_selectors import smoke_identity\n"
            f"root = Path({tmp!r})\n"
            f"cfg = yaml.safe_load((root / {config!r}).read_text())\n"
            "print(json.dumps(smoke_identity(cfg, root=root)))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONPATH": str(Path(tmp) / "src"), "PATH": "/usr/bin:/bin"},
        ).stdout
    return json.loads(out)


def validate(bundle: Path) -> dict:
    m = json.loads((bundle / "bundle_manifest.json").read_text())
    problems: list[str] = []
    for name, want in m["files_sha256"].items():
        got = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if got != want:
            problems.append(f"hash mismatch: {name}")
    rec = json.loads((bundle / MARKER).read_text())
    if rec.get("passed") is not True or rec.get("status") != "PASS":
        problems.append("record does not say passed/PASS")
    if rec.get("failures"):
        problems.append(f"record lists failures: {rec['failures']}")
    again = smoke_failures(rec.get("records") or [])
    if again:
        problems.append(f"re-applied checks fail: {again[:3]}")
    want_id = _identity_at(m["producing_commit"], m["config"])
    if rec.get("identity") != want_id:
        diff = sorted(
            k for k in want_id if (rec.get("identity") or {}).get(k) != want_id[k]
        )
        problems.append(f"identity differs from the producing commit's: {diff}")
    kinds = collections.Counter(r["kind"] for r in rec.get("records") or [])
    de = m["design_executed"]
    for k in ("replay", "identity", "generation"):
        if kinds.get(k, 0) != de[k]:
            problems.append(f"{k}: {kinds.get(k, 0)} records, manifest says {de[k]}")
    t = m["timing"]
    start = datetime.datetime.fromisoformat(t["start"])
    end = datetime.datetime.fromisoformat(t["end"])
    hours = (end - start).total_seconds() / 3600 * int(t["gpus"])
    if end <= start or abs(hours - float(t["allocated_gpu_hours"])) > 1e-3:
        problems.append("timing block inconsistent")
    return {
        "bundle": str(bundle),
        "verdict": "VALID" if not problems else "INVALID",
        "problems": problems,
        "records": dict(kinds),
        "allocated_gpu_hours": t["allocated_gpu_hours"],
        "producing_commit": m["producing_commit"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"command": "validate_smoke_bundle", "bundle": str(a.bundle)}))
        return 0
    v = validate(a.bundle)
    print(json.dumps(v, indent=2))
    return 0 if v["verdict"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())

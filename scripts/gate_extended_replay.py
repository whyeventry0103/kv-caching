#!/usr/bin/env python3
"""Instrument gate for the v2 extended replay emitter (Amendment P.4).

The new emitter must reproduce the validated instrument before its new measures
are trusted. On every (press, seed, problem) shared with the existing v1
profiles, the v2 TV series must match v1 within tolerance. That gives the new
code a correctness check against a gate-passed predecessor rather than trust.

Refuses to pass if there is no overlap: "nothing to compare" must not read the
same as "everything matched".

    python scripts/gate_extended_replay.py --v1 <run>/profiles --v2 <run>/profiles \\
        --tol 1e-6 --out gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[(path.parent.name, path.stem)] = payload
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", required=True, type=Path)
    p.add_argument("--v2", required=True, type=Path)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    v1, v2 = _index(a.v1), _index(a.v2)
    shared = sorted(set(v1) & set(v2))
    failures: list[str] = []
    worst = 0.0
    compared_steps = 0

    for key in shared:
        one, two = v1[key], v2[key]
        if two.get("schema_version") != "tv-profile-v2":
            failures.append(f"{key}: v2 side is {two.get('schema_version')!r}")
            continue
        if one.get("schema_version") != "tv-profile-v1":
            failures.append(f"{key}: v1 side is {one.get('schema_version')!r}")
            continue
        a_est, b_est = one["estimate"], two["estimate"]
        if len(a_est) != len(b_est):
            failures.append(f"{key}: length {len(a_est)} vs {len(b_est)}")
            continue
        for i, (x, y) in enumerate(zip(a_est, b_est, strict=True)):
            delta = abs(float(x) - float(y))
            worst = max(worst, delta)
            compared_steps += 1
            if delta > a.tol:
                failures.append(f"{key}: step {i} TV {x} vs {y} (delta {delta:.3g})")
                break
        # the new measures must actually be present and well-formed
        for field in ("jsd", "top1_agree", "rank_displacement"):
            if len(two.get(field, [])) != len(b_est):
                failures.append(f"{key}: {field} absent or wrong length")

    payload = {
        "amendment": "P.4",
        "n_v1": len(v1),
        "n_v2": len(v2),
        "n_shared": len(shared),
        "compared_steps": compared_steps,
        "worst_tv_delta": worst,
        "tolerance": a.tol,
        "failures": failures[:20],
        "n_failures": len(failures),
        "passed": bool(shared) and not failures,
    }
    if not shared:
        payload["passed"] = False
        payload["failures"] = ["no shared (press,problem) between v1 and v2"]
        payload["n_failures"] = 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    if not payload["passed"]:
        print("GATE FAILED — do not read the extended measures", file=sys.stderr)
        return 1
    print(f"GATE PASSED — v2 reproduces v1 TV on {compared_steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

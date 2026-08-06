#!/usr/bin/env python3
"""Executable week-2 gate for the RSP pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(summary: dict[str, float]) -> list[str]:
    failures = []
    if summary.get("fullcache_accuracy", 0.0) < 0.8:
        failures.append("full-cache accuracy must be at least 0.80")
    if summary.get("truncation_rate", 1.0) > 0.05:
        failures.append("truncation rate must be at most 0.05")
    if summary.get("compression_effect", 0.0) < 0.05:
        failures.append("compression effect must be at least 0.05")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    failures = validate(json.loads(args.summary.read_text(encoding="utf-8")))
    if failures:
        print("\n".join(f"FAIL: {failure}" for failure in failures))
        return 1
    print("PASS: RSP week-2 gate met all three criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kvcache.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli("bank", "Build full-cache reference trace banks"))

#!/usr/bin/env python3
"""Forward readout under a protected-selector family (execution plan WP4/WP5;
prepared 2026-10-05, HELD).

``scripts/run_replay.py`` is not modified. As ``run_replay_pp.py`` does, this
runner validates the config, points ``run_replay`` at the protected-selector
press factory in place of ``build_replay_press``, and runs its ``main``
unchanged. Reverse readout needs no protected press (the reader is the full
cache) and runs through ``run_replay.py`` itself. Real execution is refused
while the config is held or before the smoke PASS marker exists.

    python scripts/run_replay_protected.py \\
        --config configs/experiments/readout_math500_protsel256_wp5.yaml --dry-run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_replay

from kvcache.cli import load_config
from kvcache.presses.protected_selectors import (
    build_protected_selector_press,
    require_protected_config,
)


def build_replay_press_protected(name: str, **kwargs: Any) -> Any:
    """``build_replay_press``'s contract: always the instrumented press."""
    return build_protected_selector_press(name, instrumented=True, **kwargs)


def main() -> int:
    args = run_replay._parser().parse_args()
    config = load_config(args.config)
    require_protected_config(
        config,
        dry_run=bool(getattr(args, "dry_run", False)),
        runs_root=os.environ.get("REMOTE_RUNS_ROOT"),
        replay=True,
    )
    run_replay.build_replay_press = build_replay_press_protected
    return run_replay.main()


if __name__ == "__main__":
    raise SystemExit(main())

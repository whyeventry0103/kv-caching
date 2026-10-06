#!/usr/bin/env python3
"""Free-running grid under a protected-selector family (execution plan WP4/WP5;
prepared 2026-10-05, HELD).

``scripts/run_grid.py`` is not modified: its bytes are generation identity for
finished cells. Like ``run_grid_pp.py``, this runner validates the config,
points ``run_grid`` at ``build_protected_selector_press`` in place of
``build_press``, and runs its ``main`` unchanged. Real execution is refused
while the config is held or before the smoke PASS marker exists
(``require_protected_config``); dry runs are always allowed.

    python scripts/run_grid_protected.py \\
        --config configs/experiments/grid_math500_protsel256_wp5.yaml --dry-run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_grid

from kvcache.cli import load_config
from kvcache.presses.protected_selectors import (
    build_protected_selector_press,
    needs_eager_attention,
    require_protected_config,
)


def main() -> int:
    args = run_grid._parser().parse_args()
    config = load_config(args.config)
    require_protected_config(
        config,
        dry_run=bool(getattr(args, "dry_run", False)),
        runs_root=os.environ.get("KVCACHE_BASE_RUNS_ROOT"),
    )
    run_grid.build_press = build_protected_selector_press
    run_grid.press_needs_eager_attention = needs_eager_attention
    return run_grid.main()


if __name__ == "__main__":
    raise SystemExit(main())

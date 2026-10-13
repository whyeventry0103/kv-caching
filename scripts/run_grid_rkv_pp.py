#!/usr/bin/env python3
"""Free-running grid under prompt-protected R-KV, ``rkv_pp_v1`` (Meeting 5,
M5-A; prepared 2026-10-14, HELD).

``scripts/run_grid.py`` is not modified: its bytes are generation identity for
finished cells. Like ``run_grid_protected.py``, this runner validates the
config, points ``run_grid`` at ``build_rkv_prompt_protected_press`` in place of
``build_press``, and runs its ``main`` unchanged. Real execution is refused
while the config is held or without a current rkv-pp smoke PASS
(``require_rkv_pp_config``); dry runs are always allowed.

    python scripts/run_grid_rkv_pp.py \\
        --config configs/experiments/grid_math500_rkvpp256_m5.yaml --dry-run
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
from kvcache.presses.rkv_prompt_protect import (
    build_rkv_prompt_protected_press,
    needs_eager_attention,
    require_rkv_pp_config,
)


def main() -> int:
    args = run_grid._parser().parse_args()
    config = load_config(args.config)
    require_rkv_pp_config(
        config,
        dry_run=bool(getattr(args, "dry_run", False)),
        runs_root=os.environ.get("KVCACHE_BASE_RUNS_ROOT"),
    )
    run_grid.build_press = build_rkv_prompt_protected_press
    run_grid.press_needs_eager_attention = needs_eager_attention
    return run_grid.main()


if __name__ == "__main__":
    raise SystemExit(main())

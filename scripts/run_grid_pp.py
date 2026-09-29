#!/usr/bin/env python3
"""Free-running grid under a prompt-protected press (Amendment S.1).

``scripts/run_grid.py`` is not modified: its bytes are generation identity for
finished cells. This runner validates that the config is a registered
prompt-protected cell, points ``run_grid`` at
``kvcache.presses.prompt_protect.build_prompt_protected_press`` in place of
``build_press``, and runs its ``main`` unchanged, so the generation path,
resume logic, guards and row schema are the unprotected grid's own. Rows carry
the press names ``random_pp`` / ``streamingllm_pp``.

    python scripts/run_grid_pp.py \\
        --config configs/experiments/grid_math500_pp256_s1.yaml --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_grid

from kvcache.cli import load_config
from kvcache.presses.prompt_protect import (
    build_prompt_protected_press,
    require_pp_config,
)


def main() -> int:
    args = run_grid._parser().parse_args()
    config = load_config(args.config)
    require_pp_config(config)
    run_grid.build_press = build_prompt_protected_press
    return run_grid.main()


if __name__ == "__main__":
    raise SystemExit(main())

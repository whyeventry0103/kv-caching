#!/usr/bin/env python3
"""Forward readout under prompt-protected R-KV, ``rkv_pp_v1`` (Meeting 5, M5-A;
prepared 2026-10-14, HELD).

``scripts/run_replay.py`` is not modified. As ``run_replay_protected.py`` does,
this runner validates the config, points ``run_replay`` at the rkv_pp_v1 press
factory in place of ``build_replay_press``, and runs its ``main`` unchanged.
Reverse readout needs no protected press (the reader is the full cache) and runs
through ``run_replay.py`` itself. Real execution is refused while the config is
held or without a current rkv-pp smoke PASS.

    python scripts/run_replay_rkv_pp.py \\
        --config configs/experiments/readout_math500_rkvpp256_m5.yaml --dry-run
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
from kvcache.presses.rkv_prompt_protect import (
    build_rkv_prompt_protected_press,
    needs_eager_attention,
    require_rkv_pp_config,
)


def build_replay_press_rkv_pp(name: str, **kwargs: Any) -> Any:
    """``build_replay_press``'s contract: always the instrumented press."""
    return build_rkv_prompt_protected_press(name, instrumented=True, **kwargs)


def main() -> int:
    args = run_replay._parser().parse_args()
    config = load_config(args.config)
    require_rkv_pp_config(
        config,
        dry_run=bool(getattr(args, "dry_run", False)),
        runs_root=os.environ.get("KVCACHE_BASE_RUNS_ROOT"),
        replay=True,
    )
    run_replay.build_replay_press = build_replay_press_rkv_pp
    run_replay.press_needs_eager_attention = needs_eager_attention
    return run_replay.main()


if __name__ == "__main__":
    raise SystemExit(main())

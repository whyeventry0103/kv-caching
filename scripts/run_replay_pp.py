#!/usr/bin/env python3
"""Forward readout under a prompt-protected press (Amendment S.1).

``scripts/run_replay.py`` is not modified. This runner validates that the
config is a registered prompt-protected readout, points ``run_replay`` at the
prompt-protected press factory in place of ``build_replay_press``, and runs its
``main`` unchanged. Reverse readout needs no protected press (the reader is
the full cache) and runs through ``run_replay.py`` itself.

    python scripts/run_replay_pp.py \\
        --config configs/experiments/readout_math500_pp256_s1.yaml --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_replay

from kvcache.cli import load_config
from kvcache.presses.prompt_protect import (
    build_prompt_protected_press,
    require_pp_config,
)


def build_replay_press_pp(name: str, **kwargs: Any) -> Any:
    """``build_replay_press``'s contract: always the instrumented press."""
    return build_prompt_protected_press(name, instrumented=True, **kwargs)


def main() -> int:
    args = run_replay._parser().parse_args()
    config = load_config(args.config)
    require_pp_config(config, replay=True)
    run_replay.build_replay_press = build_replay_press_pp
    return run_replay.main()


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared experiment CLI planning and remote-execution guards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--i-understand",
        action="store_true",
        help="confirm this remote command may launch costly GPU work",
    )
    return result


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing config: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required; install pinned requirements") from exc
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"config must be a YAML mapping: {path}")
    required = {"name", "models", "benchmarks", "presses", "budgets", "seeds"}
    missing = sorted(required - config.keys())
    if missing:
        raise SystemExit(f"config missing keys: {', '.join(missing)}")
    return config


def plan(config: dict[str, Any], config_path: Path, command: str) -> dict[str, Any]:
    result = {
        "command": command,
        "config": str(config_path),
        "name": config["name"],
        "models": config["models"],
        "benchmarks": config["benchmarks"],
        "presses": config["presses"],
        "budgets": config["budgets"],
        "seeds": config["seeds"],
        "estimated_generation_count": config.get("estimated_generation_count", 0),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_cli(command: str, description: str) -> int:
    args = parser(description).parse_args()
    config = load_config(args.config)
    plan(config, args.config, command)
    if args.dry_run:
        print("DRY RUN: validated only; no model, dataset, CUDA, or job launch")
        return 0
    if not args.i_understand:
        raise SystemExit(
            "REFUSED: remote execution may be costly; rerun with --i-understand"
        )
    if os.environ.get("KVCACHE_REMOTE") != "1":
        raise SystemExit("REFUSED: non-dry execution requires KVCACHE_REMOTE=1")
    raise SystemExit(
        f"{command} remote adapter is scaffolded but not implemented; "
        "verify pinned backends before launching"
    )

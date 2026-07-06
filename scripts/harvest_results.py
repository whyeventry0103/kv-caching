#!/usr/bin/env python3
"""Harvest completed queue results into report artifacts + a digest (remote, CPU).

One command turns "runs finished overnight" into "numbers ready for the paper":
for every analysis UNIT whose required runs are complete and which has not been
harvested yet, run the right report (report_m4 / report_add / profile summary),
save outputs under <runs>/harvest/<unit>/, and print a compact digest of the
numbers the paper needs. Already-harvested units are skipped (delete the unit's
harvest dir to force a re-run); incomplete units print their progress.

Morning workflow (see RESULTS_PLAYBOOK.md for what each number means and where
it goes in the paper):

    ssh kvcache-gpu 'cd ~/kv-caching && .venv/bin/python scripts/harvest_results.py'

--dry-run prints unit readiness without running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RUNS = Path(
    os.environ.get("KVCACHE_BASE_RUNS_ROOT", "/home/sk10741/kvcache-storage/runs")
)
FC = "math500-fc-powered-merged"


@dataclass(frozen=True)
class Unit:
    name: str
    # run_name -> (kind, expected_rows); kind in {grid, readout} (grid counts the
    # bench subtree, readout counts readout/). bench applies to grid-kind counting.
    requires: dict[str, tuple[str, int]]
    kind: str  # m4 | m4_reverse | add | replay_profiles | bank_acc
    bench: str = "math500"
    args: dict[str, str] = field(default_factory=dict)


UNITS = [
    Unit(
        "m4-2048",
        {"grid-diss2048-tok": ("grid", 800), "readout-diss2048": ("readout", 800)},
        "m4",
        args={"compression": "grid-diss2048-tok", "readout": "readout-diss2048"},
    ),
    Unit(
        "add-2048",
        {"grid-diss2048-tok": ("grid", 800)},
        "add",
        args={"compression": "grid-diss2048-tok"},
    ),
    Unit(
        "m4-2048-h2o",
        {"grid-diss2048-h2o": ("grid", 400), "readout-diss2048-h2o": ("readout", 400)},
        "m4",
        args={"compression": "grid-diss2048-h2o", "readout": "readout-diss2048-h2o"},
    ),
    Unit(
        "add-256-h2o",
        {"grid-diss256-h2o": ("grid", 400)},
        "add",
        args={"compression": "grid-diss256-h2o"},
    ),
    Unit(
        "replay-256-h2o",
        {"replay-diss256-h2o": ("grid", 400)},
        "replay_profiles",
        args={"replay": "replay-diss256-h2o"},
    ),
    Unit(
        "m4-rkv-256",
        {
            "grid-diss256-rkv": ("grid", 400),
            "readout-diss256-rkv": ("readout", 400),
            "reverse-readout-diss256-rkv": ("readout", 400),
        },
        "m4_reverse",
        args={
            "compression": "grid-diss256-rkv",
            "readout": "readout-diss256-rkv",
            "reverse": "reverse-readout-diss256-rkv",
        },
    ),
    Unit(
        "m4-llama-256",
        {
            "fc-llama50": ("grid", 200),
            "grid-diss256-llama": ("grid", 400),
            "readout-diss256-llama": ("readout", 400),
            "reverse-readout-diss256-llama": ("readout", 400),
        },
        "m4_reverse",
        args={
            "fullcache": "fc-llama50",
            "compression": "grid-diss256-llama",
            "readout": "readout-diss256-llama",
            "reverse": "reverse-readout-diss256-llama",
        },
    ),
    Unit("fc-aime24-acc", {"fc-aime24": ("grid", 240)}, "bank_acc", bench="aime24"),
    Unit(
        "m4-aime-2048",
        {
            "fc-aime24": ("grid", 240),
            "grid-aime24-2048": ("grid", 360),
            "readout-aime24-2048": ("readout", 360),
        },
        "m4",
        bench="aime24",
        args={
            "fullcache": "fc-aime24",
            "compression": "grid-aime24-2048",
            "readout": "readout-aime24-2048",
        },
    ),
]


def count_rows(run_name: str, kind: str, bench: str) -> int:
    base = RUNS / run_name / ("readout" if kind == "readout" else bench)
    if not base.exists():
        return 0
    total = 0
    for path in base.rglob("*.jsonl"):
        if "token_ids" in path.parts:
            continue
        total += sum(
            1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
    return total


def _run(argv: list[str]) -> str:
    result = subprocess.run(
        argv, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{argv[2]} failed:\n{result.stdout[-800:]}{result.stderr[-800:]}"
        )
    return result.stdout


def _digest_m4(summary_path: Path) -> list[str]:
    summary = json.loads(summary_path.read_text())
    lines = [f"  fc_accuracy={summary['full_cache']['accuracy']:.4f}"]
    for c in summary["cells"]:
        lines.append(
            f"  {c['press']}@{c['budget']}: free={c['compressed_accuracy']:.4f} "
            f"{c['compressed_accuracy_ci95']} readout={c['readout_accuracy']:.4f} "
            f"{c['readout_accuracy_ci95']} residual={c['unexplained_residual']:.4f} "
            f"CI={c['unexplained_residual_ci95']} bracket={c['residual_bracket']} "
            f"revCI={c['reverse_residual_ci95']} trunc={c['truncation_rate']:.3f} "
            f"undef={c['n_readout_undefined']} n_rev={c['n_reverse_readout']} "
            f"flip={c['signed_flip_mass']}"
        )
    return lines


def harvest(unit: Unit, python: str) -> list[str]:
    out_dir = RUNS / "harvest" / unit.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if unit.kind in {"m4", "m4_reverse"}:
        argv = [
            python,
            "scripts/report_m4.py",
            "--fullcache-root",
            str(RUNS / unit.args.get("fullcache", FC)),
            "--compression-root",
            str(RUNS / unit.args["compression"]),
            "--readout-root",
            str(RUNS / unit.args["readout"]),
            "--out",
            str(out_dir),
            "--bench",
            unit.bench,
        ]
        if unit.kind == "m4_reverse":
            argv += ["--reverse-readout-root", str(RUNS / unit.args["reverse"])]
        _run(argv)
        return _digest_m4(out_dir / "summary.json")
    if unit.kind == "add":
        stdout = _run(
            [
                python,
                "scripts/report_add.py",
                "--fullcache-root",
                str(RUNS / FC),
                "--compression-root",
                str(RUNS / unit.args["compression"]),
                "--out",
                str(out_dir / "add.json"),
            ]
        )
        (out_dir / "stdout.json").write_text(stdout, encoding="utf-8")
        # keep the digest to the headline aggregates when parseable
        try:
            data = json.loads(stdout)
            return [f"  {json.dumps(data)[:600]}"]
        except json.JSONDecodeError:
            return [f"  {stdout[:600]}"]
    if unit.kind == "replay_profiles":
        replay_run = RUNS / unit.args["replay"]
        rows = []
        for path in (replay_run / unit.bench).rglob("*.jsonl"):
            if "token_ids" in path.parts:
                continue
            for ln in path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rows.append(json.loads(ln))
        mean_auc = statistics.fmean(r["auc_tv"] for r in rows)
        onset = statistics.median(
            r["onset_step"] for r in rows if r["onset_step"] is not None
        )
        payload = {"n": len(rows), "mean_auc_tv": mean_auc, "median_onset": onset}
        (out_dir / "auc.json").write_text(json.dumps(payload, indent=2))
        windows = _run(
            [
                python,
                "scripts/make_paper_figures.py",
                "--out-dir",
                str(out_dir),
                "--profiles-dir",
                str(replay_run / "profiles"),
                "--print-windows",
            ]
        )
        (out_dir / "windows.txt").write_text(windows, encoding="utf-8")
        window_lines = [
            f"  {ln.strip()}"
            for ln in windows.splitlines()
            if ":" in ln and "wrote" not in ln
        ]
        return [f"  {json.dumps(payload)}"] + window_lines[:4]
    if unit.kind == "bank_acc":
        rows = []
        for path in (RUNS / next(iter(unit.requires)) / unit.bench).rglob("*.jsonl"):
            if "token_ids" in path.parts:
                continue
            for ln in path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rows.append(json.loads(ln))
        n = len(rows)
        acc = sum(r["correct"] for r in rows) / n
        trunc = sum(r["truncated"] for r in rows) / n
        toks = statistics.fmean(r["n_tokens"] for r in rows)
        payload = {"rows": n, "accuracy": acc, "truncation": trunc, "mean_tokens": toks}
        (out_dir / "bank.json").write_text(json.dumps(payload, indent=2))
        return [f"  {json.dumps(payload)}"]
    raise ValueError(f"unknown unit kind: {unit.kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="readiness only")
    parser.add_argument(
        "--force", default="", help="re-harvest this unit name even if marker exists"
    )
    args = parser.parse_args()
    python = sys.executable
    ready, pending, done = [], [], []
    for unit in UNITS:
        marker = RUNS / "harvest" / unit.name / "harvested.marker"
        counts = {
            rn: (count_rows(rn, kind, unit.bench), want)
            for rn, (kind, want) in unit.requires.items()
        }
        complete = all(have >= want for have, want in counts.values())
        if marker.exists() and args.force != unit.name:
            done.append(unit.name)
            continue
        if not complete:
            missing = {rn: f"{h}/{w}" for rn, (h, w) in counts.items() if h < w}
            pending.append((unit.name, missing))
            continue
        ready.append(unit)

    for name, missing in pending:
        print(f"NOT READY {name}: {missing}")
    for name in done:
        print(f"ALREADY HARVESTED {name} (delete harvest dir or --force to redo)")
    for unit in ready:
        if args.dry_run:
            print(f"READY (dry-run, skipped): {unit.name}")
            continue
        print(f"=== HARVEST {unit.name} ===")
        try:
            for line in harvest(unit, python):
                print(line)
            (RUNS / "harvest" / unit.name / "harvested.marker").write_text("done\n")
        except Exception as exc:  # noqa: BLE001 -- report and continue to next unit
            print(f"  FAILED: {exc}")
    if not ready and not args.dry_run:
        print("(nothing new to harvest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

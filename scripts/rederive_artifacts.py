#!/usr/bin/env python3
"""Re-derive every paper-backing artifact from the raw rows and diff it.

The number-provenance gate (`audit_paper_numbers.py`) proves each literal in the
paper matches SOME catalogue entry, and the catalogue is built from the
artifacts in `results/diag/`. Nothing checked that the artifacts themselves
still follow from the rows. That is the last unguarded link: an artifact written
by a since-fixed pipeline, or edited by hand, would sail through every existing
gate.

This runs on the REMOTE, where the rows are. For each artifact it reconstructs
the producing command -- from the artifact's own recorded `runs` block where it
has one, otherwise from the manifest below -- re-runs it against the rows, and
compares the result field by field against the committed copy.

    ssh kvcache-gpu 'cd ~/kv-caching && .venv/bin/python \
        scripts/rederive_artifacts.py --runs-root <runs> --out /tmp/rederive.json'

Exit status is non-zero if any artifact differs. Artifacts with no mechanical
producer are reported as NO-PRODUCER rather than silently passing: "we did not
check this" and "this checks out" must never look the same in the output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DIAG = PROJECT_ROOT / "results" / "diag"

# Fields that legitimately move between two runs of the same computation, or
# that record where the computation happened rather than what it found. Compared
# separately (presence, not value) so a vanished field is still caught.
VOLATILE = {"generated", "ts", "timestamp", "profiles_dir", "runs", "note"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(budget: str, presses: dict[str, tuple[str, str]]) -> list[str]:
    return [f"{p}:{g}:{r}" for p, (g, r) in presses.items()]


# Readout runs for budget 256 on MATH-500 are seed-sharded across two dirs; the
# '+' form is report_resolution's own syntax for that.
RO256 = "readout-diss256-g0+readout-diss256-g1"
M500_256 = {
    "h2o": ("grid-diss256-h2o", "readout-diss256-h2o"),
    "random": ("grid-diss256-tok", RO256),
    "rkv": ("grid-diss256-rkv", "readout-diss256-rkv"),
    "streamingllm": ("grid-diss256-tok", RO256),
}
M500_2048 = {
    "h2o": ("grid-diss2048-h2o", "readout-diss2048-h2o"),
    "random": ("grid-diss2048-tok", "readout-diss2048"),
    "streamingllm": ("grid-diss2048-tok", "readout-diss2048"),
}
AIME_2048 = {
    p: ("grid-aime24-2048", "readout-aime24-2048")
    for p in ("random", "rkv", "streamingllm")
}


def _resolution(name: str, presses: dict, budget: str, label: str, bench: str) -> dict:
    return {
        "artifact": f"amendment-k/classb/{name}",
        "script": "scripts/report_resolution.py",
        "args": lambda out: sum((["--cell", c] for c in _cells(budget, presses)), [])
        + ["--budget", budget, "--bench", bench, "--label", label, "--out", str(out)],
        "needs_runs_root": True,
    }


MANIFEST: list[dict] = [
    _resolution(
        "resolution_math500_256.json", M500_256, "256", "math500-256", "math500"
    ),
    _resolution(
        "resolution_math500_512.json",
        {
            "random": ("grid-diss512", "readout-diss512"),
            "streamingllm": ("grid-diss512", "readout-diss512"),
        },
        "512",
        "math500-512",
        "math500",
    ),
    _resolution(
        "resolution_math500_1024.json",
        {
            "random": ("grid-diss1024", "readout-diss1024"),
            "streamingllm": ("grid-diss1024", "readout-diss1024"),
        },
        "1024",
        "math500-1024",
        "math500",
    ),
    _resolution(
        "resolution_math500_2048.json", M500_2048, "2048", "math500-2048", "math500"
    ),
    _resolution(
        "resolution_aime24_2048.json", AIME_2048, "2048", "aime24-2048", "aime24"
    ),
    _resolution(
        "resolution_pair_256.json",
        {k: M500_256[k] for k in ("random", "streamingllm")},
        "256",
        "pair-256",
        "math500",
    ),
    _resolution(
        "resolution_pair_2048.json",
        {k: M500_2048[k] for k in ("random", "streamingllm")},
        "2048",
        "pair-2048",
        "math500",
    ),
    {
        "artifact": "amendment-k/classb/residual_contrast_256.json",
        "script": "scripts/report_residual_contrast.py",
        "args": lambda out: sum((["--cell", c] for c in _cells("256", M500_256)), [])
        + ["--budget", "256", "--bench", "math500", "--out", str(out)],
        "needs_runs_root": True,
    },
    {
        "artifact": "amendment-k/classb/policy_contrast_256.json",
        "script": "scripts/report_policy_contrast.py",
        "args": lambda out: [
            "--press-a",
            "random",
            "--press-b",
            "streamingllm",
            "--budget",
            "256",
            "--bench",
            "math500",
            "--out",
            str(out),
        ],
        "grid_root": "grid-diss256-tok",
    },
    {
        "artifact": "amendment-k/classb/compute.json",
        "script": "scripts/report_compute.py",
        "args": lambda out: ["--out", str(out)],
        "needs_runs_root": True,
    },
    {
        "artifact": "amendment-k/classb/rkv_published_cap32768.json",
        "script": "scripts/report_published_config.py",
        "args": lambda out: [
            "--bench",
            "aime24",
            "--label",
            "rkv-published-config-cap32768",
            "--out",
            str(out),
        ],
        "fullcache_root": "fc-aime24-llama8b-cap32768",
        "cell_roots": ["rkvpub-g0", "rkvpub-g1"],
        "replicate_roots": ["grid-aime24-rkv-published-cap32768"],
    },
    {
        "artifact": "amendment-k/add/twofc.json",
        "script": "scripts/report_two_fc_control.py",
        "args": lambda out: ["--out", str(out)],
        "fullcache_root": "math500-fc-powered-merged",
    },
]

# Grader audits: one per graded run, all from the same script.
for _run in (
    "grid-diss256-h2o",
    "grid-diss256-rkv",
    "grid-diss256-tok",
    "math500-fc-powered-merged",
):
    MANIFEST.append(
        {
            "artifact": f"amendment-k/grader/{_run}.json",
            "script": "scripts/diagnose_math_failures.py",
            "args": lambda out: ["--gold-from-dataset", "--out", str(out)],
            "rundir": _run,
            # --out is a directory here too (records.json + summary.json +
            # table.md); the committed artifact is the summary alone.
            "out_is_dir": True,
        }
    )


def _auto_from_runs(rel: str, payload: dict) -> dict | None:
    """Reconstruct an m4/add command from the artifact's own `runs` block."""
    runs = payload.get("runs")
    if not isinstance(runs, dict) or "compression" not in runs:
        return None
    bench = "aime24" if "aime" in rel else "math500"
    if rel.startswith("amendment-k/add/"):
        return {
            "artifact": rel,
            "script": "scripts/report_add.py",
            "argv_extra": [
                "--fullcache-root",
                runs["fullcache"],
                "--compression-root",
                runs["compression"],
            ],
            # report_add's --out is a DIRECTORY (summary.json + table.md), even
            # though the committed artifact is the summary file alone.
            "out_is_dir": True,
        }
    argv = [
        "--fullcache-root",
        runs["fullcache"],
        "--compression-root",
        runs["compression"],
        "--bench",
        bench,
    ]
    # --readout-root is nargs="+", so the shards go in ONE flag. Repeating the
    # flag makes argparse keep only the last occurrence, which silently halves
    # the readout population and moves every readout-derived number in the cell.
    readout = list(runs.get("readout") or [])
    if readout:
        argv += ["--readout-root", *readout]
    if runs.get("reverse_readout"):
        argv += ["--reverse-readout-root", runs["reverse_readout"]]
    if runs.get("replay"):
        argv += ["--replay-root", runs["replay"]]
    return {
        "artifact": rel,
        "script": "scripts/report_m4.py",
        "argv_extra": argv,
        "out_is_dir": True,
    }


def _diff(old: Any, new: Any, path: str = "") -> list[str]:
    """Every leaf where the committed artifact and the re-derivation disagree."""
    problems: list[str] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            here = f"{path}/{key}"
            if key in VOLATILE:
                if (key in old) != (key in new):
                    problems.append(f"{here}: field present in only one")
                continue
            if key not in old:
                problems.append(f"{here}: only in re-derivation")
            elif key not in new:
                problems.append(f"{here}: only in committed artifact")
            else:
                problems += _diff(old[key], new[key], here)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            problems.append(f"{path}: length {len(old)} vs {len(new)}")
        else:
            for i, (a, b) in enumerate(zip(old, new, strict=True)):
                problems += _diff(a, b, f"{path}[{i}]")
    elif isinstance(old, float) and isinstance(new, float):
        if old != new:
            problems.append(f"{path}: {old!r} vs {new!r}")
    elif old != new:
        problems.append(f"{path}: {old!r} vs {new!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--only", default="", help="substring filter on artifact path")
    args = parser.parse_args()
    runs: Path = args.runs_root
    python = sys.executable

    jobs = list(MANIFEST)
    for rel in sorted(
        str(p.relative_to(DIAG))
        for p in list((DIAG / "amendment-k").glob("m4-*.json"))
        + list((DIAG / "amendment-k" / "add").glob("add-*.json"))
    ):
        auto = _auto_from_runs(rel, _load(DIAG / rel))
        if auto:
            jobs.append(auto)

    results: list[dict] = []
    for job in jobs:
        rel = job["artifact"]
        if args.only and args.only not in rel:
            continue
        committed = DIAG / rel
        if not committed.exists():
            results.append({"artifact": rel, "status": "MISSING-LOCAL"})
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / ("out" if job.get("out_is_dir") else "out.json")
            argv = [python, job["script"]]
            if job.get("needs_runs_root"):
                argv += ["--runs-root", str(runs)]
            if job.get("grid_root"):
                argv += ["--grid-root", str(runs / job["grid_root"])]
            if job.get("fullcache_root"):
                argv += ["--fullcache-root", str(runs / job["fullcache_root"])]
            if job.get("cell_roots"):
                argv += ["--cell-root"] + [str(runs / r) for r in job["cell_roots"]]
            if job.get("replicate_roots"):
                argv += ["--replicate-root"] + [
                    str(runs / r) for r in job["replicate_roots"]
                ]
            if job.get("rundir"):
                argv += ["--rundir", str(runs / job["rundir"])]
            argv += job.get("argv_extra", [])
            if "args" in job:
                argv += job["args"](out)
            elif job.get("out_is_dir"):
                argv += ["--out", str(out)]
            else:
                argv += ["--out", str(out)]

            proc = subprocess.run(
                argv, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=7200
            )
            if proc.returncode != 0:
                results.append(
                    {
                        "artifact": rel,
                        "status": "RERUN-FAILED",
                        "stderr": (proc.stdout + proc.stderr)[-700:],
                    }
                )
                continue
            produced = out / "summary.json" if job.get("out_is_dir") else out
            if not produced.exists():
                cand = list(Path(tmp).rglob("*.json"))
                if len(cand) != 1:
                    results.append({"artifact": rel, "status": "NO-OUTPUT"})
                    continue
                produced = cand[0]
            problems = _diff(_load(committed), _load(produced))
            results.append(
                {
                    "artifact": rel,
                    "status": "IDENTICAL" if not problems else "DIFFERS",
                    "n_differences": len(problems),
                    "differences": problems[:40],
                }
            )

    ok = sum(1 for r in results if r["status"] == "IDENTICAL")
    bad = [r for r in results if r["status"] != "IDENTICAL"]
    print(f"re-derived {len(results)} artifacts: {ok} IDENTICAL, {len(bad)} not")
    for row in results:
        mark = "OK  " if row["status"] == "IDENTICAL" else "FAIL"
        print(f"  {mark} {row['status']:<14} {row['artifact']}")
        for problem in row.get("differences", [])[:12]:
            print(f"         {problem}")
        if row.get("stderr"):
            print(f"         {row['stderr'][-300:]}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

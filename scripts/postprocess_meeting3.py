#!/usr/bin/env python3
"""R.14 postprocessing stage: validated read -> figures -> manifest -> marker.

Run by ``nightly_autorun.run_cpu_stages`` on every tick once the stage's
generation jobs are complete (CPU only; never allocated a GPU). Each stage
runs its registered reader with ``--final``, draws its figure only if that read
is FINAL, refreshes the Meeting #3 manifest, and writes
``<runs>/postprocess/<stage>.json`` atomically:

* ``terminal: true`` when the read is FINAL (``passed: true``) or INVALID
  (``passed: false`` -- a failed control or broken coverage needs a human, and
  re-reading will not change it);
* ``terminal: false`` for PARTIAL, so the next tick retries.

Idempotent: rerunning a finished stage reproduces the same outputs from the
same rows. Outputs land in ``results/diag/meeting3/`` and ``figures/`` of the
remote tree; committing them to the repository is a separate, logged step.

    python scripts/postprocess_meeting3.py --stage capture-b256 --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kvcache.provenance import write_json_atomic

OUT = "results/diag/meeting3"
EXP = "configs/experiments"
STAGES: dict[str, dict] = {
    "capture-b256": {
        "reader": [
            "scripts/report_capture.py",
            "--run",
            "capture-b256",
            "--config",
            f"{EXP}/capture_math500_b256_r9.yaml",
            "--out",
            f"{OUT}/capture_b256.json",
        ],
        "output": f"{OUT}/capture_b256.json",
        "figure": ("capture", f"{OUT}/capture_b256.json"),
    },
    "capture-b2048": {
        "reader": [
            "scripts/report_capture.py",
            "--run",
            "capture-b2048",
            "--config",
            f"{EXP}/capture_math500_b2048_r9.yaml",
            "--out",
            f"{OUT}/capture_b2048.json",
        ],
        "output": f"{OUT}/capture_b2048.json",
        "figure": ("capture", f"{OUT}/capture_b2048.json"),
    },
    "dacc": {
        "reader": [
            "scripts/report_dacc.py",
            "--arms",
            "0=dacc-t0",
            "512=dacc-t512",
            "2048=dacc-t2048",
            "--legacy",
            "256=grid-diss256-tok",
            "2048=grid-diss2048-tok",
            "--design",
            f"{EXP}/grid_math500_dacc_t0_r3.yaml",
            "--out",
            f"{OUT}/dacc.json",
        ],
        "output": f"{OUT}/dacc.json",
        "figure": ("dacc", f"{OUT}/dacc.json"),
    },
    "exact-lag": {
        "reader": [
            "scripts/report_exact_lag.py",
            "--run",
            "exact-lag-r5",
            "--design",
            f"{EXP}/exact_lag_math500_r5.yaml",
            "--out",
            f"{OUT}/exact_lag.json",
        ],
        "output": f"{OUT}/exact_lag.json",
        "figure": ("exact_lag", f"{OUT}/exact_lag.json"),
    },
    "fork-control": {
        "reader": [
            "scripts/report_fork_control.py",
            "--run",
            "fork-control-random",
            "--run",
            "fork-control-slm",
            "--design",
            f"{EXP}/fork_control_random_r11.yaml",
            "--design",
            f"{EXP}/fork_control_slm_r11.yaml",
            "--out",
            f"{OUT}/fork_control.json",
        ],
        "output": f"{OUT}/fork_control.json",
        "figure": None,
    },
    "t1-tv": {
        "reader": [
            "scripts/report_budget_curve.py",
            "--tv-runs",
            "512=t1-tv512",
            "1024=t1-tv1024",
            "--out",
            f"{OUT}/t1_budget_curve.json",
        ],
        "output": f"{OUT}/t1_budget_curve.json",
        "figure": ("t1", f"{OUT}/t1_budget_curve.json"),
    },
}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _status(stage: str, out: Path, rc: int) -> str:
    if not out.exists():
        return "PARTIAL"
    d = json.loads(out.read_text(encoding="utf-8"))
    if stage == "t1-tv":
        # the budget-curve reader has no FINAL flag: done when both TV cells
        # contributed an AUC with a CI
        cells = [c for c in d["cells"] if c["budget"] in (512, 1024)]
        ok = cells and all(c.get("auc_tv_ci95") for c in cells)
        return "FINAL" if (rc == 0 and ok) else "PARTIAL"
    if stage == "fork-control":
        cov = d.get("coverage", {}).get("status")
        if d.get("overall") == "FAIL":
            return "INVALID"
        return "FINAL" if cov == "COMPLETE" else "PARTIAL"
    return d.get("status", "PARTIAL")


def run_stage(
    stage: str,
    runs_root: Path,
    *,
    python: str = sys.executable,
    out_root: Path = PROJECT_ROOT,
) -> dict:
    """``out_root`` receives results and figures (the project tree in
    production; a temporary tree in tests, so tests never touch real artifacts)."""
    spec = STAGES[stage]
    reader_args = list(spec["reader"][1:])
    i = reader_args.index("--out")
    reader_args[i + 1] = str(out_root / reader_args[i + 1])
    argv = [
        python,
        str(PROJECT_ROOT / spec["reader"][0]),
        "--runs-root",
        str(runs_root),
        *reader_args,
    ]
    if stage != "t1-tv":
        argv.append("--final")
    res = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True)
    out = out_root / spec["output"]
    status = _status(stage, out, res.returncode)
    figures: list[str] = []
    if status == "FINAL" and spec["figure"]:
        kind, src = spec["figure"]
        if kind == "t1":
            import make_paper_figures as mpf

            if out_root == PROJECT_ROOT:
                mpf.fig8_budget_curve(PROJECT_ROOT / "figures")
                figures = [
                    "figures/fig8_budget_curve.pdf",
                    "figures/fig8_budget_curve.png",
                ]
        else:
            import make_meeting3_figures as m3f

            fn = {
                "capture": m3f.fig_capture,
                "dacc": m3f.fig_dacc,
                "exact_lag": m3f.fig_exact_lag,
            }[kind]
            made = fn(out_root / src, out_root / "figures")
            figures = [str(p.relative_to(out_root)) for p in made]
    outputs = [spec["output"], *figures]
    tsv = out.with_suffix(".conditions.tsv")
    if tsv.exists():
        outputs.append(str(tsv.relative_to(out_root)))
    marker = {
        "stage": stage,
        "status": status,
        "terminal": status in ("FINAL", "INVALID"),
        "passed": status == "FINAL",
        "reader_exit": res.returncode,
        "reader_tail": res.stdout.strip().splitlines()[-6:],
        "reader_stderr_tail": res.stderr.strip().splitlines()[-6:],
        "outputs": {p: _sha(out_root / p) for p in outputs},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(runs_root / "postprocess" / f"{stage}.json", marker)
    # The same record inside the results tree, where the manifest derives cell
    # statuses from evidence rather than from the ledger's declarations.
    write_json_atomic(out_root / OUT / "stages" / f"{stage}.json", marker)
    manifest_rc = None  # refreshed after the stage record exists
    if out_root == PROJECT_ROOT:
        manifest_rc = subprocess.run(
            [python, "scripts/meeting3_manifest.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).returncode
    marker["manifest_exit"] = manifest_rc
    return marker


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="", help="ignored (launcher compat)")
    ap.add_argument("--resume", action="store_true", help="ignored")
    ap.add_argument("--seeds", default="", help="ignored")
    ap.add_argument("--i-understand", action="store_true", help="ignored")
    a = ap.parse_args()
    if a.dry_run:
        print(json.dumps({"stage": a.stage, **STAGES[a.stage]}, indent=2))
        return 0
    runs_root = Path(os.environ["KVCACHE_BASE_RUNS_ROOT"])
    m = run_stage(a.stage, runs_root)
    print(json.dumps({k: m[k] for k in ("stage", "status", "terminal", "reader_exit")}))
    return 0 if m["terminal"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

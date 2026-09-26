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


# R.16: every input a stage's reader consumes. (run, subtree): None = the
# whole run directory. Repo inputs are committed files the reader reads as
# data. Reuse of a finished stage is valid only while ALL of these, and the
# analysis code (the reader plus every kvcache module it imports, transitively),
# are unchanged.
STAGE_DATA: dict[str, tuple[tuple[str, str | None], ...]] = {
    "capture-b256": (("capture-b256", None), ("q6-basis-fit", None)),
    "capture-b2048": (("capture-b2048", None), ("q6-basis-fit", None)),
    "dacc": (
        ("dacc-t0", None),
        ("dacc-t512", None),
        ("dacc-t2048", None),
        ("math500-fc-powered-merged", "math500"),  # full-cache comparators
        ("grid-diss256-tok", "token_ids"),  # R.8.3 determinism control
        ("grid-diss2048-tok", "token_ids"),
    ),
    "exact-lag": (("exact-lag-r5", None),),
    "fork-control": (("fork-control-random", None), ("fork-control-slm", None)),
    "t1-tv": (
        ("t1-tv512", None),
        ("t1-tv1024", None),
        ("math500-fc-powered-merged", "math500"),
        ("grid-diss256-tok", "math500"),
        ("grid-diss512", "math500"),
        ("grid-diss1024", "math500"),
        ("grid-diss2048-tok", "math500"),
    ),
}
STAGE_REPO_INPUTS: dict[str, tuple[str, ...]] = {
    "t1-tv": (
        "results/diag/amendment-k/m4-powered-diss256-summary.json",
        "results/diag/amendment-k/m4-512-summary.json",
        "results/diag/amendment-k/m4-1024-summary.json",
        "results/diag/amendment-k/m4-2048-summary.json",
        "results/diag/meeting3/prompt_lengths_budget_curve.json",
        "results/diag/amendment-p/auc_resid_tv.json",
    ),
}
STAGE_RUNS = {k: tuple(sorted({r for r, _ in v})) for k, v in STAGE_DATA.items()}
ANALYSIS_COMMON = (
    "scripts/postprocess_meeting3.py",
    "scripts/make_meeting3_figures.py",
    "scripts/meeting3_manifest.py",
    "configs/meeting3/q2_subset_pids.json",
    "configs/meeting3/input_manifest.json",
)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def kvcache_closure(files: set[str], root: Path = PROJECT_ROOT) -> set[str]:
    """``files`` plus every src/kvcache module they import, transitively."""
    import ast

    out, todo = set(), list(files)
    while todo:
        rel = todo.pop()
        if rel in out:
            continue
        out.add(rel)
        path = root / rel
        if not rel.endswith(".py") or not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                if n != "kvcache" and not n.startswith("kvcache."):
                    continue
                parts = n.split(".")
                for cand in (
                    "src/" + "/".join(parts) + ".py",
                    "src/" + "/".join(parts) + "/__init__.py",
                ):
                    if (root / cand).exists() and cand not in out:
                        todo.append(cand)
    return out


def analysis_files(stage: str, root: Path = PROJECT_ROOT) -> list[str]:
    spec = STAGES[stage]
    files = set(ANALYSIS_COMMON) | {spec["reader"][0]}
    files |= {a for a in spec["reader"] if a.startswith("configs/")}
    if stage == "t1-tv":
        files |= {"scripts/make_paper_figures.py"}
    return sorted(kvcache_closure(files, root))


def analysis_id(stage: str, root: Path = PROJECT_ROOT) -> str:
    """sha256 over the reader, its transitive kvcache imports, shared analysis
    code and the stage's configs."""
    h = hashlib.sha256()
    for rel in analysis_files(stage, root):
        h.update(rel.encode() + b"\0")
        h.update((root / rel).read_bytes() if (root / rel).exists() else b"")
    return h.hexdigest()


def generation_digest(stage: str, runs_root: Path, root: Path = PROJECT_ROOT) -> str:
    """sha256 over every data input the stage's reader consumes (R.16)."""
    h = hashlib.sha256()
    for run, sub in STAGE_DATA[stage]:
        base = Path(runs_root) / run / (sub or "")
        if not base.exists():
            h.update(f"{run}/{sub}:absent".encode())
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix in (".jsonl", ".json", ".npz"):
                h.update(str(f.relative_to(runs_root)).encode() + b"\0")
                h.update(hashlib.sha256(f.read_bytes()).digest())
    for rel in STAGE_REPO_INPUTS.get(stage, ()):
        h.update(f"repo:{rel}".encode() + b"\0")
        p = root / rel
        h.update(hashlib.sha256(p.read_bytes()).digest() if p.exists() else b"absent")
    return h.hexdigest()


def fresh_status(stage: str, report: dict | None, rc: int) -> tuple[str, str | None]:
    """(status, gate outcome) from THIS attempt's report and exit code only.

    A reader that crashed (no report, or an exit code other than 0 = FINAL and
    3 = not final) is READER-FAILED, retried; an old FINAL file is never read.
    Fork: analysis completion (FINAL) and the control's outcome are separate.
    """
    if report is None or rc not in (0, 3):
        return "READER-FAILED", None
    if stage == "t1-tv":
        cells = [c for c in report.get("cells", []) if c.get("budget") in (512, 1024)]
        ok = rc == 0 and len(cells) == 4 and all(c.get("auc_tv_ci95") for c in cells)
        return ("FINAL" if ok else "PARTIAL"), None
    st = report.get("status", "PARTIAL")
    if st == "FINAL" and rc != 0:
        return "READER-FAILED", None  # a FINAL report must come with exit 0
    gate = report.get("overall") if stage == "fork-control" else None
    return st, gate


def _publish(src: Path, dst: Path, history: Path) -> None:
    """Move an existing ``dst`` into history (kept), then move ``src`` in."""
    if dst.exists():
        history.mkdir(parents=True, exist_ok=True)
        os.replace(dst, history / dst.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def run_stage(
    stage: str,
    runs_root: Path,
    *,
    python: str = sys.executable,
    out_root: Path = PROJECT_ROOT,
    manifest_argv: list[str] | None = None,
) -> dict:
    """One transactional attempt (R.15). ``out_root`` receives results and
    figures (the project tree in production, a temporary tree in tests)."""
    spec = STAGES[stage]
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt = out_root / OUT / "attempts" / f"{stage}-{ts}-{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=True)
    final_json = out_root / spec["output"]
    fresh_json = attempt / final_json.name
    reader_args = list(spec["reader"][1:])
    reader_args[reader_args.index("--out") + 1] = str(fresh_json)
    argv = [
        python,
        str(PROJECT_ROOT / spec["reader"][0]),
        "--runs-root",
        str(runs_root),
        *reader_args,
    ]
    if stage != "t1-tv":
        argv.append("--final")
    digest_before = generation_digest(stage, runs_root)
    res = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True, text=True)
    try:
        report = json.loads(fresh_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    status, gate = fresh_status(stage, report, res.returncode)
    record: dict = {
        "stage": stage,
        "status": status,
        "gate_outcome": gate,
        "attempt": str(attempt.relative_to(out_root)),
        "reader_exit": res.returncode,
        "reader_tail": res.stdout.strip().splitlines()[-6:],
        "reader_stderr_tail": res.stderr.strip().splitlines()[-6:],
        "analysis_id": analysis_id(stage),
        "generation_digest": digest_before,
        "finalized": False,
        "terminal": False,
        "passed": False,
        "timestamp": ts,
    }
    outputs: list[str] = []
    if status in ("FINAL", "INVALID"):
        figures: list[Path] = []
        if status == "FINAL" and spec["figure"]:
            kind, _src = spec["figure"]
            if kind == "t1":
                import make_paper_figures as mpf

                mpf.fig8_budget_curve(attempt, t1_path=fresh_json)
                figures = [
                    attempt / "fig8_budget_curve.pdf",
                    attempt / "fig8_budget_curve.png",
                ]
            else:
                import make_meeting3_figures as m3f

                fn = {
                    "capture": m3f.fig_capture,
                    "dacc": m3f.fig_dacc,
                    "exact_lag": m3f.fig_exact_lag,
                }[kind]
                figures = fn(fresh_json, attempt)
                if not figures:
                    record["status"] = status = "FIGURE-FAILED"
        if status in ("FINAL", "INVALID"):
            hist = out_root / OUT / "history" / f"{stage}-{ts}"
            if fresh_json != final_json:
                _publish(fresh_json, final_json, hist)
            outputs.append(str(final_json.relative_to(out_root)))
            tsv = attempt / final_json.with_suffix(".conditions.tsv").name
            if tsv.exists():
                dst = final_json.with_suffix(".conditions.tsv")
                _publish(tsv, dst, hist)
                outputs.append(str(dst.relative_to(out_root)))
            for f in figures:
                # never into the source tree: the preflight hashes it, and a
                # generated file there halts every later launch (R.16.6)
                dst = out_root / OUT / "figures" / f.name
                _publish(f, dst, hist)
                outputs.append(str(dst.relative_to(out_root)))
    record["outputs"] = {p: _sha(out_root / p) for p in outputs}
    # Phase 1: the stage record exists (not finalized) so the manifest can
    # derive the cell status; phase 2 only after the manifest refresh succeeds.
    # `.stage.json`: a bare `<stage>.json` collides with a reader output of the
    # same name (dacc.json), and cited basenames must be unique (R.16.7)
    stage_record = out_root / OUT / "stages" / f"{stage}.stage.json"
    write_json_atomic(stage_record, record)
    if status in ("FINAL", "INVALID"):
        margv = manifest_argv or (
            [python, "scripts/meeting3_manifest.py"]
            if out_root == PROJECT_ROOT
            else None
        )
        mrc = (
            subprocess.run(
                margv, cwd=PROJECT_ROOT, capture_output=True, text=True
            ).returncode
            if margv
            else 0
        )
        record["manifest_exit"] = mrc
        if mrc == 0 and all(record["outputs"].values()):
            record.update(finalized=True, terminal=True, passed=status == "FINAL")
        else:
            record["status"] = "MANIFEST-FAILED" if mrc else "OUTPUT-MISSING"
    write_json_atomic(stage_record, record)
    write_json_atomic(Path(runs_root) / "postprocess" / f"{stage}.json", record)
    return record


def stage_done(stage: str, runs_root: Path, *, out_root: Path = PROJECT_ROOT) -> bool:
    """Scheduler acceptance: terminal marker AND it still describes reality.

    The outputs it lists must exist with the recorded hashes, the analysis code
    and configs must be unchanged, and the stage's input rows must be unchanged
    since the read. Otherwise the stage is re-run (idempotently).
    """
    try:
        m = json.loads(
            (Path(runs_root) / "postprocess" / f"{stage}.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return False
    if m.get("terminal") is not True or m.get("finalized") is not True:
        return False
    if m.get("analysis_id") != analysis_id(stage):
        return False
    if m.get("generation_digest") != generation_digest(stage, runs_root):
        return False
    outs = m.get("outputs") or {}
    return bool(outs) and all(_sha(out_root / p) == h for p, h in outs.items())


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
    print(
        json.dumps(
            {
                k: m.get(k)
                for k in ("stage", "status", "terminal", "reader_exit", "gate_outcome")
            }
        )
    )
    return 0 if m["terminal"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

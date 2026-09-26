"""R.15 acceptance: all six postprocessing stages through their REAL reader
subprocesses, with manifest refresh, completion markers and scheduler-side
acceptance (``stage_done``), plus the failure and restart paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import meeting3_fixtures as fx  # noqa: E402
import postprocess_meeting3 as pp  # noqa: E402

FIGURES = {
    "capture-b256": ["results/diag/meeting3/figures/meeting3_capture_b256.png"],
    "capture-b2048": ["results/diag/meeting3/figures/meeting3_capture_b2048.png"],
    "dacc": ["results/diag/meeting3/figures/meeting3_dacc.png"],
    "exact-lag": ["results/diag/meeting3/figures/meeting3_exact_lag.png"],
    "fork-control": [],
    "t1-tv": ["results/diag/meeting3/figures/fig8_budget_curve.png"],
}
BUILD = {
    "capture-b256": lambda r: fx.capture(r, 256),
    "capture-b2048": lambda r: fx.capture(r, 2048),
    "dacc": fx.dacc,
    "exact-lag": fx.exact_lag,
    "fork-control": fx.fork,
    "t1-tv": fx.t1_tv,
}


OUTPUTS = {
    "capture-b256": [
        "results/diag/meeting3/capture_b256.json",
        "results/diag/meeting3/figures/meeting3_capture_b256.pdf",
    ],
    "capture-b2048": [
        "results/diag/meeting3/capture_b2048.json",
        "results/diag/meeting3/figures/meeting3_capture_b2048.pdf",
    ],
    "dacc": [
        "results/diag/meeting3/dacc.json",
        "results/diag/meeting3/figures/meeting3_dacc.pdf",
    ],
    "exact-lag": [
        "results/diag/meeting3/exact_lag.json",
        "results/diag/meeting3/figures/meeting3_exact_lag.pdf",
    ],
    "fork-control": ["results/diag/meeting3/fork_control.json"],
    "t1-tv": [
        "results/diag/meeting3/t1_budget_curve.json",
        "results/diag/meeting3/figures/fig8_budget_curve.pdf",
    ],
}


def _real_manifest(stage, tmp_path):
    """The REAL manifest builder over a minimal isolated ledger (R.16)."""
    out = tmp_path / "out"
    cells = tmp_path / "cells.yaml"
    cells.write_text(
        json.dumps(
            {
                "model": "m",
                "reference_bank": "b",
                "runtime_pin": {},
                "cells": [
                    {
                        "id": f"cell-{stage}",
                        "task": 1,
                        "protocol": "R.16",
                        "stage": stage,
                        "status": "QUEUED",
                        "outputs": OUTPUTS[stage],
                    }
                ],
            }
        )
    )
    return [
        sys.executable,
        str(ROOT / "scripts" / "meeting3_manifest.py"),
        "--root",
        str(out),
        "--cells",
        str(cells),
        "--out",
        str(out / "manifest.json"),
    ]


def _manifest_status(tmp_path, stage):
    d = json.loads((tmp_path / "out" / "manifest.json").read_text())
    return {c["id"]: c["status"] for c in d["cells"]}[f"cell-{stage}"]


def _run(stage, tmp_path, **kw):
    runs, out = tmp_path / "runs", tmp_path / "out"
    kw.setdefault("manifest_argv", _real_manifest(stage, tmp_path))
    return pp.run_stage(stage, runs, out_root=out, **kw), runs, out


@pytest.mark.parametrize("stage", sorted(pp.STAGES))
def test_every_stage_finalizes_through_its_real_reader(stage, tmp_path):
    BUILD[stage](tmp_path / "runs")
    m, runs, out = _run(stage, tmp_path)
    assert m["reader_exit"] == 0, m["reader_stderr_tail"]
    assert m["status"] == "FINAL", m
    assert m["terminal"] and m["finalized"] and m["passed"]
    for f in FIGURES[stage]:
        assert (out / f).exists(), f
    assert all(h for h in m["outputs"].values())
    rec = json.loads(
        (out / "results/diag/meeting3/stages" / f"{stage}.stage.json").read_text()
    )
    assert rec["finalized"] is True and rec["manifest_exit"] == 0
    assert pp.stage_done(stage, runs, out_root=out)
    # the real manifest derived the ledger status from this stage's evidence
    assert _manifest_status(tmp_path, stage) == "COMPLETE"
    if stage == "fork-control":
        assert m["gate_outcome"] == "PASS"
    if stage == "dacc":
        d = json.loads((out / "results/diag/meeting3/dacc.json").read_text())
        assert d["determinism_control"]["status"] == "PASSED"


def test_an_old_final_followed_by_a_reader_failure_is_not_accepted(tmp_path):
    fx.capture(tmp_path / "runs")
    m, runs, out = _run("capture-b256", tmp_path)
    final = out / "results/diag/meeting3/capture_b256.json"
    before = final.read_bytes()
    broken = tmp_path / "broken_python"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    m2, _, _ = _run("capture-b256", tmp_path, python=str(broken))
    assert m2["status"] == "READER-FAILED" and not m2["terminal"]
    assert final.read_bytes() == before  # the old report is untouched, not re-accepted
    assert not pp.stage_done("capture-b256", runs, out_root=out)  # retried next tick


def test_manifest_refresh_failure_is_retryable_and_never_terminal(tmp_path):
    fx.capture(tmp_path / "runs")
    m, runs, out = _run("capture-b256", tmp_path, manifest_argv=["false"])
    assert m["status"] == "MANIFEST-FAILED" and not m["terminal"]
    assert (
        out / "results/diag/meeting3/figures/meeting3_capture_b256.png"
    ).exists()  # drawn, not accepted
    assert not pp.stage_done("capture-b256", runs, out_root=out)
    m2, _, _ = _run("capture-b256", tmp_path)
    assert m2["terminal"] and pp.stage_done("capture-b256", runs, out_root=out)
    assert list((out / "results/diag/meeting3/history").iterdir())  # old kept


def test_changed_rows_or_analysis_code_invalidate_acceptance(tmp_path, monkeypatch):
    fx.exact_lag(tmp_path / "runs")
    m, runs, out = _run("exact-lag", tmp_path)
    assert pp.stage_done("exact-lag", runs, out_root=out)
    monkeypatch.setattr(pp, "analysis_id", lambda s: "changed-code")
    assert not pp.stage_done("exact-lag", runs, out_root=out)
    monkeypatch.undo()
    with (runs / "exact-lag-r5/math500/m/rows.jsonl").open("a") as fh:
        fh.write("\n")
    assert not pp.stage_done("exact-lag", runs, out_root=out)


@pytest.mark.parametrize(
    "stage,build",
    [
        ("capture-b256", lambda r: fx.capture(r, 256, bad_control=True)),
        ("exact-lag", lambda r: fx.exact_lag(r, drop_lag=True)),
        ("exact-lag", lambda r: fx.exact_lag(r, bad_gate=True)),
        ("fork-control", lambda r: fx.fork(r, infinite_control=True)),
    ],
    ids=[
        "capture-failed-control",
        "lag-payload-missing",
        "restoration-gate-false",
        "fork-infinite-control",
    ],
)
def test_invalid_evidence_is_terminal_invalid_without_a_verdict(stage, build, tmp_path):
    build(tmp_path / "runs")
    m, runs, out = _run(stage, tmp_path)
    assert m["status"] == "INVALID" and m["terminal"] and not m["passed"]
    for f in FIGURES[stage]:
        assert not (out / f).exists()
    assert _manifest_status(tmp_path, stage) == "INVALID"


def test_a_missing_condition_stays_partial_and_is_retried(tmp_path):
    fx.capture(tmp_path / "runs")
    f = tmp_path / "runs/capture-b256/math500/m/rows.jsonl"
    f.write_text("".join(f.read_text().splitlines(keepends=True)[1:]))
    m, runs, out = _run("capture-b256", tmp_path)
    assert m["status"] == "PARTIAL" and not m["terminal"]
    assert not pp.stage_done("capture-b256", runs, out_root=out)


# ---------------------------------------------------------------- R.16 reuse
def _mutate_jsonl(path, fn):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_changed_full_cache_comparator_invalidates_dacc(tmp_path):
    """Review of f68d3f1: the FC bank was outside the digest; FINAL stayed accepted."""
    fx.dacc(tmp_path / "runs")
    m, runs, out = _run("dacc", tmp_path)
    assert pp.stage_done("dacc", runs, out_root=out)
    _mutate_jsonl(
        runs / "math500-fc-powered-merged/math500/m/fc.jsonl",
        lambda rows: rows[0].update(correct=False),
    )
    assert not pp.stage_done("dacc", runs, out_root=out)


def test_a_changed_legacy_determinism_token_invalidates_dacc(tmp_path):
    fx.dacc(tmp_path / "runs")
    m, runs, out = _run("dacc", tmp_path)
    _mutate_jsonl(
        runs / "grid-diss256-tok/token_ids/m/t.jsonl",
        lambda rows: rows[0].update(trace_token_ids=[9, 9]),
    )
    assert not pp.stage_done("dacc", runs, out_root=out)


def test_a_changed_historical_tv_input_invalidates_t1(tmp_path):
    fx.t1_tv(tmp_path / "runs")
    m, runs, out = _run("t1-tv", tmp_path)
    assert pp.stage_done("t1-tv", runs, out_root=out)
    _mutate_jsonl(
        runs / "grid-diss512/math500/m/rows.jsonl",
        lambda rows: rows[0].update(correct=not rows[0]["correct"]),
    )
    assert not pp.stage_done("t1-tv", runs, out_root=out)


def test_a_changed_imported_estimator_changes_the_analysis_identity(tmp_path):
    import shutil

    for stage, module in (
        ("dacc", "src/kvcache/stats.py"),
        ("exact-lag", "src/kvcache/metrics/restoration.py"),
    ):
        root = tmp_path / stage
        for rel in pp.analysis_files(stage):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        before = pp.analysis_id(stage, root)
        assert module in pp.analysis_files(stage, root)
        with (root / module).open("a") as fh:
            fh.write("\n# estimator changed\n")
        assert pp.analysis_id(stage, root) != before, stage


def test_every_kvcache_import_of_every_reader_is_fingerprinted():
    import ast

    for stage in pp.STAGES:
        files = set(pp.analysis_files(stage))
        for rel in [f for f in files if f.endswith(".py")]:
            tree = ast.parse((ROOT / rel).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "kvcache."
                ):
                    mod = "src/" + node.module.replace(".", "/")
                    assert f"{mod}.py" in files or f"{mod}/__init__.py" in files, (
                        stage,
                        rel,
                        node.module,
                    )


@pytest.mark.parametrize("stage", sorted(pp.STAGES))
def test_no_stage_writes_into_the_source_tree(stage, tmp_path):
    """R.16.6: a figure published into figures/ changed the remote source-tree
    hash, so every later preflight failed and the queue halted. Every output a
    stage publishes (and everything it writes) must be under results/."""
    BUILD[stage](tmp_path / "runs")
    m, runs, out = _run(stage, tmp_path)
    assert m["status"] == "FINAL"
    assert all(p.startswith("results/") for p in m["outputs"]), m["outputs"]
    written = [p.relative_to(out) for p in out.rglob("*") if p.is_file()]
    stray = [
        str(p) for p in written if p.parts[0] != "results" and p.name != "manifest.json"
    ]
    assert not stray, stray


@pytest.mark.parametrize("stage", sorted(pp.STAGES))
def test_stage_records_never_share_a_basename_with_published_outputs(stage, tmp_path):
    """R.16.7: the D_acc record `stages/dacc.json` shared its basename with the
    reader output `dacc.json`, so a bare citation of `dacc.json` resolved to two
    files and the results/diag uniqueness test failed on real data. Every file a
    stage publishes under results/diag must have a basename of its own."""
    BUILD[stage](tmp_path / "runs")
    m, runs, out = _run(stage, tmp_path)
    assert m["status"] == "FINAL"
    diag = out / "results" / "diag"
    names = [p.name for p in diag.rglob("*") if p.is_file()]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, dupes


def test_standalone_figure_script_defaults_outside_the_source_tree():
    """R.17 0.4(b): run by hand with no --out-dir, the figure script must not
    write into figures/ (hashed source tree) -- the path that halted the queue."""
    import make_meeting3_figures as m3f

    rel = m3f.FIG_DIR.relative_to(m3f.PROJECT_ROOT)
    assert rel.parts[0] == "results", rel

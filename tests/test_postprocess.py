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

OK_MANIFEST = [sys.executable, "-c", "pass"]
FIGURES = {
    "capture-b256": ["figures/meeting3_capture_b256.png"],
    "capture-b2048": ["figures/meeting3_capture_b2048.png"],
    "dacc": ["figures/meeting3_dacc.png"],
    "exact-lag": ["figures/meeting3_exact_lag.png"],
    "fork-control": [],
    "t1-tv": ["figures/fig8_budget_curve.png"],
}
BUILD = {
    "capture-b256": lambda r: fx.capture(r, 256),
    "capture-b2048": lambda r: fx.capture(r, 2048),
    "dacc": fx.dacc,
    "exact-lag": fx.exact_lag,
    "fork-control": fx.fork,
    "t1-tv": fx.t1_tv,
}


def _run(stage, tmp_path, **kw):
    runs, out = tmp_path / "runs", tmp_path / "out"
    return (
        pp.run_stage(stage, runs, out_root=out, **{"manifest_argv": OK_MANIFEST, **kw}),
        runs,
        out,
    )


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
        (out / "results/diag/meeting3/stages" / f"{stage}.json").read_text()
    )
    assert rec["finalized"] is True and rec["manifest_exit"] == 0
    assert pp.stage_done(stage, runs, out_root=out)
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
    assert (out / "figures/meeting3_capture_b256.png").exists()  # drawn, not accepted
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


def test_a_missing_condition_stays_partial_and_is_retried(tmp_path):
    fx.capture(tmp_path / "runs")
    f = tmp_path / "runs/capture-b256/math500/m/rows.jsonl"
    f.write_text("".join(f.read_text().splitlines(keepends=True)[1:]))
    m, runs, out = _run("capture-b256", tmp_path)
    assert m["status"] == "PARTIAL" and not m["terminal"]
    assert not pp.stage_done("capture-b256", runs, out_root=out)

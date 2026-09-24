"""Reader arithmetic for the R.9 capture read, on hand-built rows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from report_capture import paired_contrast, trigger_ratio  # noqa: E402


def _row(pid, seed, press, coll, omega, l0=0.0):
    return {
        "pid": pid,
        "seed": seed,
        "press": press,
        "budget": 256,
        "checks": {"n_nonfinite": 0, "layer0_delta_max": l0},
        "collateral_pooled": coll,
        "wanted_any_head_pooled": 0.5,
        "M_mean": 0.2,
        "M_event_mean": 0.01,
        "Delta_mean": 0.1,
        "slices": {"trigger": {"M": [[0.2, 0.4]], "Delta": [[0.1, 0.1]]}},
        "latent": {"omega_trace": omega},
    }


def test_trigger_ratio_is_mean_delta_over_mean_m():
    r = _row("a", 1, "rkv", 0.1, 1.0)
    assert trigger_ratio(r) == pytest.approx(0.1 / 0.3)
    r["slices"] = {}
    assert trigger_ratio(r) is None


def test_paired_contrast_pairs_by_problem_and_seed():
    by = {
        "rkv": {("a", 1): 0.1, ("a", 2): 0.2, ("b", 1): 0.3},
        "h2o": {("a", 1): 0.3, ("a", 2): 0.4, ("c", 1): 0.9},
    }
    c = paired_contrast(by, "rkv", "h2o", n_resamples=200, seed=0)
    assert c["n_pairs"] == 2 and c["n_problems"] == 1
    assert c["mean"] == pytest.approx(-0.2)


def test_reader_verdicts_and_control_exclusion(tmp_path):
    run = tmp_path / "cap" / "math500" / "m"
    run.mkdir(parents=True)
    rows = []
    for i, pid in enumerate("abcdef"):
        for seed in (1, 2):
            rows += [
                _row(pid, seed, "rkv", 0.10 + 0.01 * i, 1.0),
                _row(pid, seed, "h2o", 0.30 + 0.01 * i, 1.0),
                _row(pid, seed, "random", 0.2, 2.0 + 0.1 * i),
                _row(pid, seed, "streamingllm", 0.2, 1.0 + 0.1 * i),
            ]
    rows.append(_row("z", 1, "rkv", 0.9, 9.0, l0=1e-3))  # fails layer-0 control
    (run / "x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o.json"
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_capture.py"),
            "--runs-root",
            str(tmp_path),
            "--run",
            "cap",
            "--out",
            str(out),
            "--expected-per-policy",
            "12",
            "--n-resamples",
            "200",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    d = json.loads(out.read_text())
    assert d["n_rows_failing_controls"] == 1
    assert d["q3_registered_contrast"]["mean"] == pytest.approx(-0.2)
    assert d["q3_registered_contrast"]["verdict"].startswith("SUPPORTS")
    assert d["q6_registered_contrast"]["verdict"] == "PREDICTED"
    assert d["complete"] is True

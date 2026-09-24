"""End-to-end check of the D_acc reader on a hand-built run tree."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _row(pid, seed, correct, press="fullcache", budget="full", onset=None):
    r = {
        "pid": pid,
        "press": press,
        "budget": budget,
        "decoding": {"seed": seed},
        "correct": correct,
        "truncated": False,
        "n_tokens": 100,
    }
    if onset is not None:
        r["onset"] = onset
    return r


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _onset(t0, exposed=True):
    return {
        "compression_start_step": t0,
        "eviction_onset_step": 65 if exposed else None,
        "achieved_exposure": 35 if exposed else 0,
        "exposed": exposed,
    }


def _run(tmp_path, *extra):
    out = tmp_path / "dacc.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "report_dacc.py"),
        "--runs-root",
        str(tmp_path),
        "--arms",
        "0=dacc-t0",
        "--expected-per-cell",
        "2",
        "--n-resamples",
        "50",
        "--out",
        str(out),
        *extra,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res, out


def test_reads_damage_flips_and_exposure_and_marks_partial_cells(tmp_path):
    fc = tmp_path / "math500-fc-powered-merged" / "math500" / "m"
    _write(fc / "fc-s1.jsonl", [_row("a", 1, True), _row("b", 1, True)])
    arm = tmp_path / "dacc-t0" / "math500" / "m"
    _write(
        arm / "random-256-s1.jsonl",
        [
            _row("a", 1, False, "random", 256, _onset(0)),  # lost
            _row("b", 1, True, "random", 256, _onset(0, exposed=False)),
        ],
    )
    _write(
        arm / "random-2048-s1.jsonl", [_row("a", 1, True, "random", 2048, _onset(0))]
    )
    res, out = _run(tmp_path)
    assert res.returncode == 0, res.stderr
    d = json.loads(out.read_text())
    c = d["cells"]["random-256-t0"]
    assert c["status"] == "READ"
    assert c["fc_accuracy"] == 1.0 and c["comp_accuracy"] == 0.5
    assert c["damage_fc_minus_comp"]["mean"] == 0.5
    assert c["paired"]["harmful_flip_mass"]["mean"] == 0.5
    assert c["exposed_fraction"] == 0.5 and c["n_unexposed"] == 1
    assert d["cells"]["random-2048-t0"] == {"status": "PARTIAL", "n_rows": 1}


def test_refuses_rows_that_are_not_gated_d_acc_rows(tmp_path):
    fc = tmp_path / "math500-fc-powered-merged" / "math500" / "m"
    _write(fc / "fc-s1.jsonl", [_row("a", 1, True)])
    arm = tmp_path / "dacc-t0" / "math500" / "m"
    _write(arm / "random-256-s1.jsonl", [_row("a", 1, True, "random", 256)])
    res, _ = _run(tmp_path)
    assert res.returncode != 0
    assert "lacks an onset block" in res.stderr

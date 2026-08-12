"""The two refusals in scripts/report_restoration.py (Amendment P.5).

A restoration report that reads divergence from an ungated reinsertion, or that
anchors arms which landed on different steps, produces a plausible-looking
number that means nothing. Both refusals are tested here because both are the
kind that stays correct only while something checks it.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_restoration", ROOT / "scripts" / "report_restoration.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _profile(n=400, value=0.5):
    return {"estimate": [value] * n, "jsd": [value] * n}


def _build_run(root, *, gate_passed=True, applied=None, include_gate=True):
    """A two-problem restoration cell with all four arms."""
    applied = applied or {}
    base = root / "run" / "profiles" / "math500" / "model"
    for arm in ("none", "relaxed", "matched", "random"):
        cell = base / f"random.{arm}-256-s11"
        cell.mkdir(parents=True, exist_ok=True)
        for pid in ("aaa", "bbb"):
            # Treated arms drop to 0.1 from step 100; `none` stays at 0.5.
            series = _profile()
            if arm != "none":
                series["estimate"] = [0.5] * 100 + [0.1] * 300
            (cell / f"{pid}.json").write_text(json.dumps(series), encoding="utf-8")
            if not include_gate:
                continue
            report = {"passed": gate_passed, "arm": arm}
            if arm != "none":
                report["applied_at_step"] = applied.get((arm, pid), 100)
            (cell / f"{pid}.restoration.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
    return root


def _run(root, out, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_restoration.py"),
            "--runs-root",
            str(root),
            "--run",
            "run",
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_a_clean_run_reads_and_classifies(tmp_path):
    _build_run(tmp_path)
    out = tmp_path / "restoration.json"
    result = _run(tmp_path, out, "--window", "50")
    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    cell = payload["cells"]["random-256"]
    assert cell["anchored_on"] == "applied_step"
    assert cell["n_traces"] == 2
    # none (0.5) minus treated (0.1) over the anchored window.
    assert cell["arms"]["matched"]["effect_vs_no_intervention"] == pytest.approx(0.4)
    # matched and random are identical here, so the effect is NON-SPECIFIC --
    # which is exactly the verdict a "any perturbation helps" result must get.
    assert cell["verdict"] == "NON-SPECIFIC"


def test_a_failed_gate_stops_the_report(tmp_path):
    _build_run(tmp_path, gate_passed=False)
    result = _run(tmp_path, tmp_path / "o.json", "--window", "50")
    assert result.returncode != 0
    assert "INSTRUMENT GATE NOT CLEAN" in result.stderr


def test_a_missing_gate_stops_the_report(tmp_path):
    """Absence must not read as a pass."""
    _build_run(tmp_path, include_gate=False)
    result = _run(tmp_path, tmp_path / "o.json", "--window", "50")
    assert result.returncode != 0
    assert "INSTRUMENT GATE NOT CLEAN" in result.stderr


def test_arms_that_landed_on_different_steps_are_not_pooled_on_one_anchor(tmp_path):
    """One trace's arms disagree, so that trace is dropped and the fact recorded."""
    _build_run(tmp_path, applied={("matched", "aaa"): 250})
    out = tmp_path / "restoration.json"
    result = _run(tmp_path, out, "--window", "50")
    assert result.returncode == 0, result.stderr
    cell = json.loads(out.read_text(encoding="utf-8"))["cells"]["random-256"]
    assert cell["n_anchor_disagreements"] == 1
    assert cell["n_traces"] == 1

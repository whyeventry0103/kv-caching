"""R.14 D_acc read: exact grid coverage, identity, equal-problem damage."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_dacc as rd  # noqa: E402

PIDS, SEEDS, PRESSES, BUDGETS = ["a", "b"], [1, 2], ["random"], [256]


def _row(pid, seed, t0, correct, press="random", budget=256):
    return {
        "pid": pid,
        "press": press,
        "budget": budget,
        "decoding": {"seed": seed},
        "correct": correct,
        "truncated": False,
        "n_tokens": 100,
        "provenance": {"kind": "dacc", "config_sha256": f"cfg{t0}"},
        "onset": {
            "compression_start_step": t0,
            "eviction_onset_step": 65,
            "achieved_exposure": 35,
            "exposed": True,
        },
    }


def _arms():
    return {
        0: [_row(p, s, 0, p == "b") for p in PIDS for s in SEEDS],
        512: [_row(p, s, 512, True) for p in PIDS for s in SEEDS],
    }


FC = {(p, s): True for p in PIDS for s in SEEDS}


def _read(arms, fc=FC):
    return rd.read_dacc(
        arms,
        fc,
        pids=PIDS,
        seeds=SEEDS,
        presses=PRESSES,
        budgets=BUDGETS,
        n_resamples=100,
    )


def test_complete_grid_reads_equal_problem_damage():
    out = _read(_arms())
    assert out["status"] == "FINAL"
    c = out["cells"]["random-256-t0"]
    assert c["damage_fc_minus_comp"]["mean"] == pytest.approx(0.5)
    assert c["harmful_flip_mass"]["mean"] == pytest.approx(0.5)
    assert out["cells"]["random-256-t512"]["damage_fc_minus_comp"]["mean"] == 0.0
    assert len(out["table"]) == 8


def test_duplicate_standing_in_for_a_missing_seed_is_invalid():
    arms = _arms()
    arms[0] = arms[0][:-1] + [copy.deepcopy(arms[0][0])]
    out = _read(arms)
    assert out["coverage"]["status"] == "INVALID" and out["status"] != "FINAL"
    assert out["cells"]["random-256-t0"]["status"] == "PARTIAL"


def test_missing_arm_rows_or_fc_comparators_are_not_final():
    arms = _arms()
    arms[512] = arms[512][:2]
    assert _read(arms)["status"] == "PARTIAL"
    fc = dict(FC)
    fc.pop(("a", 1))
    out = _read(_arms(), fc)
    assert out["status"] != "FINAL" and out["fc_comparators_missing"]


def test_mixed_identity_within_an_arm_is_invalid():
    arms = _arms()
    arms[0][1]["provenance"]["config_sha256"] = "other"
    assert _read(arms)["status"] == "INVALID"


def test_unequal_seed_counts_use_the_equal_problem_mean():
    from kvcache.coverage import paired_equal_problem

    fc = {("A", s): 1.0 for s in range(4)} | {("B", 0): 1.0}
    comp = {("A", s): 0.0 for s in range(4)} | {("B", 0): 1.0}
    assert paired_equal_problem(fc, comp, n_resamples=50)["mean"] == 0.5


def test_cli_final_is_nonzero_without_the_determinism_control(tmp_path, monkeypatch):
    for t0, rows in _arms().items():
        d = tmp_path / f"dacc-t{t0}" / "math500" / "m"
        d.mkdir(parents=True)
        (d / "x.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fcd = tmp_path / "math500-fc-powered-merged" / "math500" / "m"
    fcd.mkdir(parents=True)
    (fcd / "fc.jsonl").write_text(
        "".join(
            json.dumps({"pid": p, "decoding": {"seed": s}, "correct": True}) + "\n"
            for p in PIDS
            for s in SEEDS
        )
    )
    monkeypatch.setattr(rd, "_design", lambda _c: (PIDS, SEEDS, PRESSES, BUDGETS))
    out = tmp_path / "o.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--runs-root",
            str(tmp_path),
            "--arms",
            "0=dacc-t0",
            "512=dacc-t512",
            "--design",
            "d.yaml",
            "--out",
            str(out),
            "--final",
            "--n-resamples",
            "50",
        ],
    )
    assert rd.main() == 3  # grid complete, but R.8.3 control not run
    d = json.loads(out.read_text())
    assert d["status"] == "PARTIAL" and "determinism" in d["why_not_final"]
    assert (tmp_path / "o.conditions.tsv").exists()

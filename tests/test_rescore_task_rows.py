"""Read-time GSM8K/BBH scoring through the real CLI (review of 6b409001, R4):
every arm re-scored from stored text with the frozen grader of the frozen
revision; grader and trace hashes recorded; MATH refused; malformed rows and
a wrong revision are INVALID; an old output is retired."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kvcache import bbh
from kvcache.task_scoring import registry

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rescore_task_rows.py"
GSM_REV = registry()["gsm8k"]["revision"]


def _row(pid, trace, press="fullcache", truncated=False, correct=False):
    return {
        "pid": pid,
        "press": press,
        "budget": 256,
        "decoding": {"seed": 11},
        "trace": trace,
        "truncated": truncated,
        "correct": correct,
    }


def _cli(tmp_path, bench, rev, gold, runs):
    gpath, out = tmp_path / "gold.json", tmp_path / "scores.json"
    gpath.write_text(json.dumps(gold))
    dirs = []
    for name, rows in runs.items():
        d = tmp_path / name / bench
        d.mkdir(parents=True, exist_ok=True)
        (d / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        dirs.append(str(tmp_path / name))
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--bench",
            bench,
            "--revision",
            rev,
            "--gold",
            str(gpath),
            "--runs",
            *dirs,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads(out.read_text())


def test_gsm8k_numbers_units_and_every_arm(tmp_path):
    gold = {"g/1": "1250", "g/2": "72", "g/3": "18"}
    full = [
        _row("g/1", r"</think> \boxed{1,250}"),
        _row("g/2", r"</think> \boxed{72 clips}"),
        _row("g/3", r"</think> \boxed{18.00}"),
    ]
    comp = [
        _row("g/1", r"</think> \boxed{6} so the answer is 7", press="random"),
        _row("g/2", "x", press="random", truncated=True),
        _row("g/3", r"</think> the answer is \$18", press="random"),
    ]
    rc, d = _cli(tmp_path, "gsm8k", GSM_REV, gold, {"fc": full, "rnd": comp})
    assert (rc, d["status"], d["n_rows"]) == (0, "FINAL", 6)
    by = {(s["press"], s["pid"]): s for s in d["scores"]}
    assert all(by[("fullcache", p)]["correct"] for p in gold)
    assert by[("random", "g/1")]["rule"] == "conflicting_final_answers"
    assert by[("random", "g/2")]["rule"] == "truncated"
    assert by[("random", "g/3")]["correct"] is True
    assert d["grader"] == "gsm8k-numeric-v2" and len(d["grader_sha256"]) == 64
    assert all(len(s["trace_sha256"]) == 64 for s in d["scores"])


def test_bbh_conflicting_boxes_and_corrections(tmp_path):
    gold = {"b/1": "(C)", "b/2": "(C)", "b/3": "(C)"}
    rows = [
        _row("b/1", r"</think> \boxed{(A)} or \boxed{(C)}"),
        _row("b/2", r"</think> \boxed{(C)}. Actually, the answer is (D)."),
        _row("b/3", r"</think> \boxed{(C)}"),
    ]
    rc, d = _cli(tmp_path, bbh.NAME, bbh.PINNED_COMMIT, gold, {"fc": rows})
    assert (rc, d["status"]) == (0, "FINAL")
    rules = {s["pid"]: (s["correct"], s["rule"]) for s in d["scores"]}
    assert rules["b/1"] == (False, "conflicting_final_answers")
    assert rules["b/2"] == (False, "conflicting_final_answers")
    assert rules["b/3"] == (True, "final_answer")


def test_refusals_are_invalid_evidence(tmp_path):
    ok = {"fc": [_row("g/1", r"\boxed{1}")]}
    rc, d = _cli(tmp_path, "math500", "x", {"g/1": "1"}, ok)
    assert (rc, d["status"]) == (2, "INVALID") and "frozen in-row grader" in d["error"]
    rc, d = _cli(tmp_path, "gsm8k", "0" * 40, {"g/1": "1"}, ok)
    assert (rc, d["status"]) == (2, "INVALID") and "not the frozen" in d["error"]
    bad = {"fc": [{**_row("g/1", r"\boxed{1}"), "truncated": "False"}]}
    rc, d = _cli(tmp_path, "gsm8k", GSM_REV, {"g/1": "1"}, bad)
    assert (rc, d["status"]) == (2, "INVALID") and not d["scores"]
    rc, d = _cli(tmp_path, "gsm8k", GSM_REV, {}, ok)
    assert d["status"] == "INVALID" and "no gold answer" in json.dumps(d["problems"])
    assert d["retired_previous_output"]  # the earlier output was retired first

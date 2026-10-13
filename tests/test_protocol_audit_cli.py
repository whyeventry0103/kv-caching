"""Acceptance tests for ``scripts/report_protocol_audit.py`` through the real CLI
(review of fbf869f, R1, 2026-10-05).

Each test builds a complete, valid fixture, checks it reads FINAL, then applies
one mutation the review showed the reader accepted:

* a grid ``correct`` field that is the string "False" or NaN (``bool()`` made
  both True) -> INVALID;
* a readout whose ``readout_defined`` / ``readout_correct`` pair is
  inconsistent -> INVALID;
* a duplicated relevant full-cache key -> INVALID (the bank is an allowed
  superset, but not a place where duplicates may collapse);
* all four Random free-run rows of one capture problem deleted -> PARTIAL with
  labels withheld (it read FINAL at 24 problems);
* a duplicated free-run key in the matched read -> INVALID;
* a stale FINAL is retired when the next read fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_s_acceptance import (  # noqa: E402
    PIDS25,
    PIDS100,
    RUNMETA,
    SEEDS,
    _write,
    capture_fixture,
    s1_fixture,
)


def _run(runs: Path, out: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_protocol_audit.py"),
            "--runs-root",
            str(runs),
            "--out-dir",
            str(out),
            "--n-resamples",
            "50",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, p.stdout + p.stderr


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------ reverse
def reverse_fixture(tmp: Path) -> Path:
    runs = s1_fixture(tmp)
    for name, presses in (
        ("reverse-readout-pp256", ["random_pp", "streamingllm_pp"]),
        ("reverse-readout-diss256", ["random", "streamingllm"]),
    ):
        rows = [
            {
                "reference_key": ["math500", p, "m", press, 256, s],
                "condition": {"press": "fullcache", "budget": 256},
                "readout_defined": p != PIDS100[98],
                "readout_correct": (
                    (PIDS100.index(p) % 3 == 0) if p != PIDS100[98] else None
                ),
                **RUNMETA,
            }
            for press in presses
            for p in PIDS100
            for s in SEEDS
        ]
        _write(runs / name / "readout/r.jsonl", rows)
    return runs


def test_reverse_final_on_a_complete_valid_fixture(tmp_path):
    runs = reverse_fixture(tmp_path)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "256")
    assert code == 0, log
    d = json.loads((tmp_path / "out/reverse_b256.json").read_text())
    assert d["status"] == "FINAL"
    assert d["per_policy"]["random_pp"]["population"]["n_common_support"] == 392


def _mutate_grid_correct(runs: Path, value) -> None:
    path = runs / "grid-pp256/math500/m/rows.jsonl"
    rows = _rows(path)
    rows[5]["correct"] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("value", ["False", float("nan"), 1], ids=["str", "nan", "int"])
def test_reverse_non_bool_correct_is_invalid(tmp_path, value):
    runs = reverse_fixture(tmp_path)
    _mutate_grid_correct(runs, value)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "256")
    assert code == 2, log
    d = json.loads((tmp_path / "out/reverse_b256.json").read_text())
    assert d["status"] == "INVALID"
    assert "correct" in json.dumps(d["population"]["grid_pp"]["malformed"])


@pytest.mark.parametrize(
    "defined,correct",
    [(True, None), (False, True), ("yes", True)],
    ids=["defined_without_value", "undefined_with_value", "flag_not_bool"],
)
def test_reverse_inconsistent_readout_flags_are_invalid(tmp_path, defined, correct):
    runs = reverse_fixture(tmp_path)
    path = runs / "reverse-readout-pp256/readout/r.jsonl"
    rows = _rows(path)
    rows[7]["readout_defined"], rows[7]["readout_correct"] = defined, correct
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "256")
    assert code == 2, log
    d = json.loads((tmp_path / "out/reverse_b256.json").read_text())
    assert d["status"] == "INVALID"


def test_reverse_duplicate_full_cache_key_is_invalid(tmp_path):
    runs = reverse_fixture(tmp_path)
    path = runs / "math500-fc-powered-merged/math500/fc.jsonl"
    rows = _rows(path)
    dup = dict(rows[0], correct=False)
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, dup]))
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "256")
    assert code == 2, log
    d = json.loads((tmp_path / "out/reverse_b256.json").read_text())
    assert d["status"] == "INVALID"
    assert d["population"]["full_cache"]["duplicated"]


def test_reverse_stale_final_is_retired_when_the_next_read_fails(tmp_path):
    runs = reverse_fixture(tmp_path)
    out = tmp_path / "out"
    assert _run(runs, out, "reverse", "--budget", "256")[0] == 0
    _mutate_grid_correct(runs, "False")
    assert _run(runs, out, "reverse", "--budget", "256")[0] == 2
    d = json.loads((out / "reverse_b256.json").read_text())
    assert d["status"] == "INVALID"
    # main() retires the old FINAL before reading, so it survives only renamed
    assert list(out.glob("reverse_b256.json.superseded-*"))
    assert not (out / "reverse_b256.conditions.tsv").exists()


# ------------------------------------------------------------------ matched
GRID_RUNS = {
    "random": "grid-diss256-tok",
    "streamingllm": "grid-diss256-tok",
    "h2o": "grid-diss256-h2o",
    "rkv": "grid-diss256-rkv",
}


def matched_fixture(tmp: Path) -> Path:
    runs = capture_fixture(tmp)
    acc = {"random": 0, "streamingllm": 2, "h2o": 3, "rkv": 1}  # correct if i % k
    by_run: dict[str, list[dict]] = {}
    for press, run in GRID_RUNS.items():
        for p in PIDS100:
            for s in SEEDS:
                by_run.setdefault(run, []).append(
                    {
                        "pid": p,
                        "press": press,
                        "budget": 256,
                        "correct": acc[press] == 0
                        or PIDS100.index(p) % acc[press] == 0,
                        "trace": "x",
                        "decoding": {"seed": s},
                        **RUNMETA,
                    }
                )
    for run, rows in by_run.items():
        _write(runs / run / "math500/m/rows.jsonl", rows)
    return runs


def test_matched_final_on_a_complete_valid_fixture(tmp_path):
    runs = matched_fixture(tmp_path)
    code, log = _run(runs, tmp_path / "out", "matched")
    assert code == 0, log
    d = json.loads((tmp_path / "out/matched_b256.json").read_text())
    assert d["status"] == "FINAL"
    assert (d["n_conditions"], d["n_problems"]) == (100, 25)
    assert d["dependencies"]["sha256"]


def test_matched_missing_accuracy_rows_are_partial_not_a_smaller_cohort(tmp_path):
    runs = matched_fixture(tmp_path)
    path = runs / "grid-diss256-tok/math500/m/rows.jsonl"
    rows = [
        r for r in _rows(path) if not (r["press"] == "random" and r["pid"] == PIDS25[3])
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    code, log = _run(runs, tmp_path / "out", "matched")
    assert code == 3, log
    d = json.loads((tmp_path / "out/matched_b256.json").read_text())
    assert d["status"] == "PARTIAL"
    assert d["accuracy_population"]["random"]["missing"]
    assert str(d["contrasts_M"]).startswith("WITHHELD")


def test_matched_duplicate_accuracy_key_is_invalid(tmp_path):
    runs = matched_fixture(tmp_path)
    path = runs / "grid-diss256-h2o/math500/m/rows.jsonl"
    rows = _rows(path)
    first = next(r for r in rows if r["pid"] == PIDS25[0])
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in [*rows, dict(first, correct=True)])
    )
    code, log = _run(runs, tmp_path / "out", "matched")
    assert code == 2, log
    d = json.loads((tmp_path / "out/matched_b256.json").read_text())
    assert d["status"] == "INVALID"


# ------------------------------------------------- B=1024 (proposal P1, held)
def reverse_fixture_1024(tmp: Path, *, drop: int = 0) -> Path:
    runs = s1_fixture(tmp, budget=1024)
    rows = [
        {
            "reference_key": ["math500", p, "m", press, 1024, s],
            "condition": {"press": "fullcache", "budget": 1024},
            "readout_defined": True,
            "readout_correct": PIDS100.index(p) % 2 == 0,
            **RUNMETA,
        }
        for press in ("random_pp", "streamingllm_pp")
        for p in PIDS100
        for s in SEEDS
    ]
    _write(runs / "reverse-readout-pp1024/readout/r.jsonl", rows[drop:])
    return runs


def test_reverse_1024_reads_the_protected_2x2_alone(tmp_path):
    runs = reverse_fixture_1024(tmp_path)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "1024")
    assert code == 0, log
    d = json.loads((tmp_path / "out/reverse_b1024.json").read_text())
    assert d["status"] == "FINAL"
    assert sorted(d["per_policy"]) == ["random_pp", "streamingllm_pp"]
    assert all(
        str(v).startswith("NOT AVAILABLE")
        for v in d["protection_by_direction"].values()
    )
    assert "reverse_base" not in d["population"]


def test_reverse_1024_incomplete_reverse_arm_is_partial(tmp_path):
    runs = reverse_fixture_1024(tmp_path, drop=3)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "1024")
    assert code == 3, log
    d = json.loads((tmp_path / "out/reverse_b1024.json").read_text())
    assert d["status"] == "PARTIAL"
    assert str(d["per_policy"]).startswith("WITHHELD")


def test_reverse_1024_carries_the_registered_holm_block(tmp_path):
    runs = reverse_fixture_1024(tmp_path)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "1024")
    assert code == 0, log
    d = json.loads((tmp_path / "out/reverse_b1024.json").read_text())
    ri = d["registered_inference"]
    assert ri["status"].startswith("REGISTERED")
    assert ri["family_size"] == 2
    assert set(ri["members"]) == {"random_pp", "streamingllm_pp"}
    for m in ri["members"].values():
        assert m["status"] in (
            "TESTED",
            "NOT TESTABLE (zero or undefined standard error)",
        )
        if m["status"] == "TESTED":
            assert m["p_holm"] >= m["p"]


def test_reverse_256_labels_the_test_exploratory(tmp_path):
    runs = reverse_fixture(tmp_path)
    code, log = _run(runs, tmp_path / "out", "reverse", "--budget", "256")
    assert code == 0, log
    d = json.loads((tmp_path / "out/reverse_b256.json").read_text())
    assert d["registered_inference"]["status"].startswith("EXPLORATORY")

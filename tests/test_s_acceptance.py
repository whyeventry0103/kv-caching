"""Acceptance tests for the Amendment S readers, through the real CLI.

Written 2026-10-02 to reproduce the five failures of the independent review of
f69dce3 BEFORE they were fixed, then kept as the production-path contract:

1. ``origin`` accepted one Random row of the 200-condition design, with no
   provenance and failed arm gates, and printed PROMPT CARRIES IT with exit 0;
2. ``determinism_control`` passed NaN against finite values, and inf vs inf;
3. ``protection`` accepted a missing undefined-readout condition replaced by a
   duplicate of another undefined one, because the row count stayed 400;
4. absent prompt-token sidecars silently produced an empty budget-relaxed
   exclusion list, so the sensitivity read became the primary read;
5. both protected accuracy channels identically zero reached the prompt-loss
   consequence branch.

Every test here runs ``scripts/report_s.py`` as a subprocess on a complete,
valid fixture and then on one mutation of it, and checks the written output,
the status and the exit code. Nothing is monkeypatched inside the reader.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvcache.gates import expected_run_identity  # noqa: E402
from kvcache.provenance import expected_inputs, load_input_manifest  # noqa: E402
from kvcache.read_validation import frozen_pids  # noqa: E402

SEEDS = (11, 23, 47, 89)
PIDS100 = frozen_pids(100, root=ROOT)
PIDS25 = frozen_pids(25, root=ROOT)
RUNMETA = {
    "runmeta": {"commit": "c" * 40, "source_tree_hash": "s" * 64},
    "versions": {"torch": "2.7.1"},
}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(runs: Path, out: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_s.py"),
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


# ------------------------------------------------------------- S.1 fixture
def s1_fixture(
    tmp: Path, *, budget: int = 256, undefined_pid: str = PIDS100[99]
) -> Path:
    """A complete, valid budget-256 S.1 cell: protected free-run right on even
    problems, unprotected never; readouts always right where defined."""
    runs = tmp / "runs"
    keys = [(p, s) for p in PIDS100 for s in SEEDS]
    _write(
        runs / "math500-fc-powered-merged/math500/fc.jsonl",
        [
            {
                "pid": p,
                "correct": True,
                "truncated": False,
                "n_tokens": 3000,
                "decoding": {"seed": s},
                **RUNMETA,
            }
            for p, s in keys
        ],
    )

    def grid(name, presses, correct):
        rows, toks = [], []
        for press in presses:
            for p, s in keys:
                rows.append(
                    {
                        "pid": p,
                        "press": press,
                        "budget": budget,
                        "correct": correct(p),
                        "trace": "x",
                        "decoding": {"seed": s},
                        **RUNMETA,
                    }
                )
                toks.append(
                    {
                        "pid": p,
                        "press": press,
                        "budget": budget,
                        "prompt_token_ids": [0] * (353 if p == PIDS100[0] else 50),
                    }
                )
        _write(runs / name / "math500/m/rows.jsonl", rows)
        _write(runs / name / "token_ids/math500/m/t.jsonl", toks)

    def readout(name, presses, correct, seeds=SEEDS):
        rows = []
        for press in presses:
            for p, s in keys:
                if s not in seeds:
                    continue
                defined = p != undefined_pid
                rows.append(
                    {
                        "reference_key": ["math500", p, "m", "fullcache", "full", s],
                        "condition": {"press": press, "budget": budget},
                        "readout_defined": defined,
                        "readout_correct": correct(p) if defined else None,
                        **RUNMETA,
                    }
                )
        _write(runs / name / "readout/r.jsonl", rows)

    even = lambda p: PIDS100.index(p) % 2 == 0  # noqa: E731
    base, pp = ["random", "streamingllm"], ["random_pp", "streamingllm_pp"]
    if budget == 256:
        grid("grid-diss256-tok", base, lambda p: False)
        grid("grid-pp256", pp, even)
        readout("readout-diss256-g0", base, lambda p: True, seeds=(11, 23))
        readout("readout-diss256-g1", base, lambda p: True, seeds=(47, 89))
        readout("readout-pp256", pp, lambda p: True)
    else:
        grid("grid-diss1024", base, lambda p: False)
        grid("grid-pp1024", pp, even)
        readout("readout-diss1024", base, lambda p: True)
        readout("readout-pp1024", pp, lambda p: True)
    return runs


def test_protection_final_on_a_complete_valid_cell(tmp_path):
    runs = s1_fixture(tmp_path)
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "256")
    assert code == 0, log
    d = json.loads((tmp_path / "out/protection_b256.json").read_text())
    assert d["status"] == "FINAL"
    assert d["budget_relaxed_problems"] == [PIDS100[0]]
    assert all(c["passed"] for c in d["controls"].values())
    blk = d["per_press"]["random_pp"]
    # 50 even problems right among the 99 defined -> residual -49/99 on the
    # defined set; unprotected residual is -1 everywhere
    assert blk["primary"]["residual_pp"]["mean"] == pytest.approx(-49 / 99)
    assert blk["primary"]["residual_label"] == "RESIDUAL PERSISTS"
    assert blk["primary"]["protection_effect"] == "SHRINKS"
    assert blk["sensitivity_without_budget_relaxed"]["n_conditions"] == 396
    assert blk["primary"]["informativeness"] == "INFORMATIVE"
    assert (tmp_path / "out/protection_b256.conditions.tsv").exists()
    assert len(d["sufficient_statistics_sha256"]) == 64


def test_review_failure_3_duplicate_undefined_key_is_invalid_not_final(tmp_path):
    runs = s1_fixture(tmp_path)
    path = runs / "readout-pp256/readout/r.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # drop the undefined condition (p99, seed 11) for random_pp and duplicate
    # another undefined one in its place: the row count stays 400
    victim = next(
        i
        for i, r in enumerate(rows)
        if r["reference_key"][1] == PIDS100[99]
        and r["reference_key"][5] == 11
        and r["condition"]["press"] == "random_pp"
    )
    other = next(
        r
        for r in rows
        if r["reference_key"][1] == PIDS100[99]
        and r["reference_key"][5] == 23
        and r["condition"]["press"] == "random_pp"
    )
    rows[victim] = dict(other)
    _write(path, rows)
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "256")
    assert code == 2, log
    d = json.loads((tmp_path / "out/protection_b256.json").read_text())
    assert d["status"] == "INVALID"
    assert d["population"]["readout_pp"]["duplicated"] == [
        ["random_pp", "256", PIDS100[99], 23]
    ]
    assert d["population"]["readout_pp"]["missing"] == [
        ["random_pp", "256", PIDS100[99], 11]
    ]
    assert d["per_press"] == {}


def test_review_failure_4_missing_prompt_metadata_fails_closed(tmp_path):
    runs = s1_fixture(tmp_path)
    (runs / "grid-pp256/token_ids/math500/m/t.jsonl").unlink()
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "256")
    assert code == 2, log
    d = json.loads((tmp_path / "out/protection_b256.json").read_text())
    assert d["status"] == "INVALID"
    assert not d["controls"]["prompt_metadata"]["passed"]
    assert d["controls"]["prompt_metadata"]["missing"]
    assert d["per_press"] == {}


def test_review_failure_5_floored_channels_are_uninformative_not_prompt_loss(tmp_path):
    runs = s1_fixture(tmp_path)
    for name in ("grid-pp256/math500/m/rows.jsonl", "readout-pp256/readout/r.jsonl"):
        path = runs / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for r in rows:
            if "correct" in r:
                r["correct"] = False
            elif r["readout_defined"]:
                r["readout_correct"] = False
        _write(path, rows)
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "256")
    assert code == 0, log
    d = json.loads((tmp_path / "out/protection_b256.json").read_text())
    for press in ("random_pp", "streamingllm_pp"):
        blk = d["per_press"][press]["primary"]
        assert blk["residual_pp"]["mean"] == 0.0
        assert blk["residual_label"].startswith("UNINFORMATIVE: floor")
    # a 1024 cell that IS informative does not rescue the consequence
    runs1024 = s1_fixture(tmp_path / "b", budget=1024)
    for name in ("grid-pp1024", "readout-pp1024", "grid-diss1024", "readout-diss1024"):
        (runs / name).symlink_to(runs1024 / name)
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "1024")
    assert code == 0, log
    code, log = _run(runs, tmp_path / "out", "consequence")
    assert code == 0, log
    c = json.loads((tmp_path / "out/consequence.json").read_text())
    assert c["consequence"].startswith("UNINFORMATIVE")
    assert "prompt loss" not in c["consequence"]


def test_partial_cell_has_no_label_and_exit_contract(tmp_path):
    runs = s1_fixture(tmp_path)
    path = runs / "readout-pp256/readout/r.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _write(path, rows[:300])
    code, log = _run(runs, tmp_path / "out", "protection", "--budget", "256")
    assert code == 3, log
    code, log = _run(
        runs, tmp_path / "out", "protection", "--budget", "256", "--partial"
    )
    assert code == 0, log
    d = json.loads((tmp_path / "out/protection_b256.json").read_text())
    assert d["status"] == "PARTIAL"
    assert (
        d["per_press"]["random_pp"]["primary"]["residual_label"] == "PARTIAL: no label"
    )
    code, log = _run(runs, tmp_path / "out", "consequence")
    assert code != 0 and "not FINAL" in log


def test_stale_final_is_retired_and_consequence_refuses_changed_inputs(tmp_path):
    runs = s1_fixture(tmp_path)
    out = tmp_path / "out"
    assert _run(runs, out, "protection", "--budget", "256")[0] == 0
    first = json.loads((out / "protection_b256.json").read_text())
    assert first["status"] == "FINAL"
    # the rows change under the FINAL output
    path = runs / "grid-pp256/math500/m/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["correct"] = not rows[0]["correct"]
    _write(path, rows)
    runs1024 = s1_fixture(tmp_path / "b", budget=1024)
    for name in ("grid-pp1024", "readout-pp1024", "grid-diss1024", "readout-diss1024"):
        (runs / name).symlink_to(runs1024 / name)
    assert _run(runs, out, "protection", "--budget", "1024")[0] == 0
    code, log = _run(runs, out, "consequence")
    assert code != 0 and "changed since it was read" in log
    # a now-broken read retires the old FINAL instead of leaving it behind
    (runs / "grid-pp256/token_ids/math500/m/t.jsonl").unlink()
    assert _run(runs, out, "protection", "--budget", "256")[0] == 2
    assert json.loads((out / "protection_b256.json").read_text())["status"] == "INVALID"
    assert list(out.glob("protection_b256.json.superseded-*"))


# ------------------------------------------------------------- S.2 fixture
def s2_fixture(tmp: Path) -> Path:
    runs = tmp / "runs"
    manifest = load_input_manifest()
    cfg3 = "configs/experiments/restore_v3_math500_s2.yaml"
    cfg2 = "configs/experiments/restore_v2_math500_r17.yaml"
    id3, _ = expected_run_identity("restore_v3", cfg3, runs, root=ROOT)
    id2, _ = expected_run_identity("restore_v2", cfg2, runs, root=ROOT)
    keys = [(p, s) for p in PIDS25 for s in SEEDS]
    _write(
        runs / "math500-fc-powered-merged/math500/fc.jsonl",
        [
            {
                "pid": p,
                "correct": True,
                "truncated": False,
                "n_tokens": 3000,
                "decoding": {"seed": s},
                **RUNMETA,
            }
            for p, s in keys
        ],
    )
    v3, v2 = [], []
    for press in ("random", "streamingllm"):
        for p, s in keys:
            inp = expected_inputs(manifest, "restore_v3", p, s)
            base = {
                "pid": p,
                "seed": s,
                "press": press,
                "budget": 256,
                "n_trace_tokens": 3000,
                "prompt_len": 50,
            }
            if s == 89 and p == PIDS25[0]:
                v3.append(
                    {
                        **base,
                        "status": "INELIGIBLE: no eviction at or after the trigger",
                        "arms": {},
                        "cohort_step": None,
                        "provenance": {**id3, "inputs_sha256": inp},
                    }
                )
                v2.append(
                    {
                        **base,
                        "status": "INELIGIBLE: no eviction at or after the trigger",
                        "provenance": {**id2, "inputs_sha256": inp},
                    }
                )
                continue
            frozen, full = 0.10, 0.02
            v3.append(
                {
                    **base,
                    "status": "READ",
                    "cohort_step": 1024,
                    "n_prompt_return": 46,
                    "none_window_mean_tv": 0.11,
                    "arms": {
                        arm: {
                            "window_mean_tv": tv,
                            "gate": {"passed": True, "pad_max": 0},
                        }
                        for arm, tv in (
                            ("frozen", frozen),
                            ("prompt", 0.04),
                            ("trace", 0.09),
                            ("full", full),
                        )
                    },
                    "provenance": {**id3, "inputs_sha256": inp},
                }
            )
            v2.append(
                {
                    **base,
                    "status": "READ",
                    "cohort_step": 1024,
                    "full_restore": {
                        "eligible": True,
                        "gate": {"passed": True},
                        "window_mean_tv": full,
                        "none_window_mean_tv": 0.11,
                    },
                    "provenance": {**id2, "inputs_sha256": inp},
                }
            )
    _write(runs / "restore-v3/math500/m/rows.jsonl", v3)
    _write(runs / "restore-v2/math500/m/rows.jsonl", v2)
    return runs


def test_origin_final_on_a_complete_valid_cell(tmp_path):
    runs = s2_fixture(tmp_path)
    code, log = _run(runs, tmp_path / "out", "origin")
    assert code == 0, log
    d = json.loads((tmp_path / "out/origin.json").read_text())
    assert d["status"] == "FINAL"
    assert d["population"]["status"] == "COMPLETE"
    assert all(c["passed"] for c in d["controls"].values()), {
        k: v for k, v in d["controls"].items() if not v["passed"]
    }
    r = d["per_policy"]["random"]["ratios"]
    assert r["label"] == "PROMPT CARRIES IT" and r["specificity"] == "PROMPT-SPECIFIC"
    assert r["rho_prompt"]["ratio"] == pytest.approx(0.75)
    assert d["per_policy"]["random"]["status_counts"] == {
        "INELIGIBLE: no eviction at or after the trigger": 1,
        "READ": 99,
    }


def test_review_failure_1_one_row_without_provenance_or_gates_is_invalid(tmp_path):
    runs = s2_fixture(tmp_path)
    path = runs / "restore-v3/math500/m/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    one = next(r for r in rows if r["press"] == "random" and r["status"] == "READ")
    one = dict(one)
    one.pop("provenance")
    one["arms"] = {a: {**b, "gate": {"passed": False}} for a, b in one["arms"].items()}
    _write(path, [one])
    code, log = _run(runs, tmp_path / "out", "origin")
    assert code == 2, log
    d = json.loads((tmp_path / "out/origin.json").read_text())
    assert d["status"] == "INVALID"
    assert d["population"]["n_rows"] == 1 and len(d["population"]["missing"]) == 199
    assert any(
        "gate not passed" in p
        for ps in d["population"]["malformed"].values()
        for p in ps
    )
    assert not d["controls"]["provenance_restore_v3"]["passed"]
    assert d["per_policy"]["random"]["ratios"].startswith("WITHHELD")
    assert "PROMPT CARRIES IT" not in json.dumps(d)
    # and --partial does not turn an INVALID read into anything else
    assert _run(runs, tmp_path / "out", "origin", "--partial")[0] == 2


def test_review_failure_2_nan_and_inf_fail_the_determinism_gate(tmp_path):
    from kvcache.metrics.s_reads import determinism_control

    v2 = {("a", 1): {"full": 0.03, "none": 0.10}}
    assert not determinism_control({("a", 1): {"full": math.nan, "none": 0.10}}, v2)[
        "passed"
    ]
    inf = {("a", 1): {"full": math.inf, "none": 0.10}}
    assert not determinism_control(inf, inf)["passed"]
    assert not determinism_control({("a", 1): {"full": 1.5, "none": 0.10}}, v2)[
        "passed"
    ]
    assert determinism_control(dict(v2), v2)["passed"]
    # and through the CLI: a NaN TV in one arm makes the read INVALID
    runs = s2_fixture(tmp_path)
    path = runs / "restore-v3/math500/m/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = next(r for r in rows if r["status"] == "READ")
    target["arms"]["full"]["window_mean_tv"] = float("nan")
    _write(path, rows)
    code, log = _run(runs, tmp_path / "out", "origin")
    assert code == 2, log
    d = json.loads((tmp_path / "out/origin.json").read_text())
    assert d["status"] == "INVALID" and d["per_policy"]["random"]["ratios"].startswith(
        "WITHHELD"
    )


def test_origin_partial_mode_works_and_withholds_labels(tmp_path):
    runs = s2_fixture(tmp_path)
    path = runs / "restore-v3/math500/m/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _write(path, rows[:120])
    assert _run(runs, tmp_path / "out", "origin")[0] == 3
    code, log = _run(runs, tmp_path / "out", "origin", "--partial")
    assert code == 0, log
    d = json.loads((tmp_path / "out/origin.json").read_text())
    assert d["status"] == "PARTIAL"
    assert all(v["ratios"].startswith("WITHHELD") for v in d["per_policy"].values())
    assert (tmp_path / "out/origin.conditions.tsv").exists()


# ------------------------------------------------------------- R.17 restore
def _run_r17(runs: Path, out: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_r17.py"),
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


def test_r17_restore_reader_has_the_same_boundary(tmp_path):
    runs = s2_fixture(tmp_path)
    code, log = _run_r17(runs, tmp_path / "out", "restore")
    assert code == 0, log
    d = json.loads((tmp_path / "out/restore_v2.json").read_text())
    assert d["status"] == "FINAL"
    assert d["validation"]["population"]["status"] == "COMPLETE"
    assert d["validation"]["controls"]["provenance_restore_v2"]["passed"]
    assert (
        d["per_policy"]["random"]["full_restore"]["branch"] == "MIXED"
    )  # R_fs = 0.02 / 0.11
    # one row without provenance: INVALID, exit 2, branch withheld
    path = runs / "restore-v2/math500/m/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[3].pop("provenance")
    _write(path, rows)
    code, log = _run_r17(runs, tmp_path / "out", "restore")
    assert code == 2, log
    d = json.loads((tmp_path / "out/restore_v2.json").read_text())
    assert d["status"] == "INVALID" and "per_policy" in d and d["per_policy"] == {}
    # a short read: PARTIAL, exit 3, branch withheld
    _write(path, [r for r in rows if "provenance" in r][:150])
    code, log = _run_r17(runs, tmp_path / "out", "restore")
    assert code == 3, log
    d = json.loads((tmp_path / "out/restore_v2.json").read_text())
    assert d["status"] == "PARTIAL"
    assert d["per_policy"]["random"]["full_restore"]["branch"].startswith("WITHHELD")

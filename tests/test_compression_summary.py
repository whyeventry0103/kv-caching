"""Nominal-budget compression summary (review of 5616630, R4): hand-computed
mean/median/ratio-1 counts, effective-budget ratios where the S.1 rule relaxes
a budget, cross-checks against recorded prompt lengths, and the real CLI
refusing to publish without FINAL, current S.1 evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kvcache import compression_summary as cs
from kvcache.compression_summary import summarize

ROOT = Path(__file__).resolve().parents[1]


def test_summary_hand_computed():
    s = summarize([0.25, 0.5, 1.0, 1.0])
    assert s["mean"] == pytest.approx(0.6875) and s["median"] == pytest.approx(0.75)
    assert s["n_ratio_one"] == 2 and s["ratio_one_share"] == 0.5
    assert "never" not in json.dumps(s)  # ratio 1 is a lower bound, not a count


@pytest.mark.parametrize("bad", [0.0, 1.5, True, "0.5", float("nan")])
def test_out_of_range_refused(bad):
    with pytest.raises(ValueError):
        summarize([bad])
    with pytest.raises(ValueError):
        summarize([])


TSV = (
    b"press\tbudget\tpid\tseed\tprompt_len\n"
    b"random_pp\t256\ta\t11\t40\n"
    b"random_pp\t256\tb\t11\t200\n"
)


def _row(pid, n, press="random_pp", budget=256, prompt=None):
    plen = {"a": 40, "b": 200}[pid] if prompt is None else prompt
    return {
        "press": press,
        "pid": pid,
        "decoding": {"seed": 11},  # the grid row schema keeps the seed here
        "budget": budget,
        "n_tokens": n,
        "compression_ratio": min(1.0, budget / (plen + n)),
    }


def test_cell_ratios_nominal_and_effective_hand_computed():
    rows = [_row("a", 984), _row("b", 100)]  # a: 256/1024; b: 256/300
    out = cs.cell_ratios(
        rows,
        press="random_pp",
        budget=256,
        keys=cs.design_keys(TSV, "random_pp"),
        prompt_len=cs.prompt_lengths(TSV),
        protected=True,
    )
    assert out["mean"] == pytest.approx((0.25 + 256 / 300) / 2)
    assert out["n_ratio_one"] == 0
    # b's prompt (200) leaves 56 < 64 trace slots: effective target 264, so
    # its effective ratio is 264/300; a is not relaxed: 256/1024 both ways
    assert out["n_budget_relaxed"] == 1
    assert out["effective"]["mean"] == pytest.approx((0.25 + 264 / 300) / 2)
    unprot = cs.cell_ratios(
        [_row("a", 984, "random"), _row("b", 56, "random")],
        press="random",
        budget=256,
        keys=cs.design_keys(TSV, "random_pp"),
        prompt_len=cs.prompt_lengths(TSV),
        protected=False,
    )
    assert unprot["n_budget_relaxed"] == 0 and "effective" not in unprot


@pytest.mark.parametrize(
    "rows, why",
    [
        ([_row("a", 984)], "do not match the S.1 design"),
        ([_row("a", 984), _row("b", 56), _row("b", 56)], "do not match"),
        ([_row("a", 984, prompt=10), _row("b", 56)], "is not min(1, B"),
        ([_row("a", 984), _row("b", 56, budget=1024)], "row budget"),
    ],
)
def test_cell_ratios_refuse_rows_the_design_does_not_vouch_for(rows, why):
    with pytest.raises(ValueError, match=why.replace("(", r"\(")):
        cs.cell_ratios(
            rows,
            press="random_pp",
            budget=256,
            keys=cs.design_keys(TSV, "random_pp"),
            prompt_len=cs.prompt_lengths(TSV),
            protected=True,
        )


def test_s1_evidence_states():
    sha = hashlib.sha256(TSV).hexdigest()
    good = {
        "status": "FINAL",
        "sufficient_statistics_sha256": sha,
        "dependencies": {"sha256": "f"},
    }
    assert cs.s1_evidence(None, None, None)[0] == "PARTIAL"
    assert cs.s1_evidence({**good, "status": "PARTIAL"}, TSV, "f")[0] == "PARTIAL"
    assert cs.s1_evidence(good, TSV, "f") == ("FINAL", [])
    assert cs.s1_evidence(good, TSV, "other")[0] == "INVALID"
    assert cs.s1_evidence(good, TSV + b"x", "f")[0] == "INVALID"


# --- the real CLI -------------------------------------------------------------


def _cli(runs, s1, out):
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_realized_compression.py"),
            "--runs-root",
            str(runs),
            "--s1-dir",
            str(s1),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads(out.read_text())


def _grid(runs, run, rows):
    p = runs / run / "math500" / "m" / "x.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_review_case_one_row_per_cell_without_evidence_publishes_nothing(tmp_path):
    """The review's reproduction: minimal rows, no prior S.1 read."""
    runs = tmp_path / "runs"
    for run, presses in [
        ("grid-pp256", ["random_pp", "streamingllm_pp"]),
        ("grid-diss256-tok", ["random", "streamingllm"]),
        ("grid-pp1024", ["random_pp", "streamingllm_pp"]),
        ("grid-diss1024", ["random", "streamingllm"]),
    ]:
        _grid(runs, run, [{"press": p, "compression_ratio": 1.0} for p in presses])
    rc, out = _cli(runs, tmp_path / "no-s1", tmp_path / "rc.json")
    assert (rc, out["status"]) == (3, "PARTIAL") and "cells" not in out


def _s1(tmp_path, runs, budget, fingerprint):
    s1 = tmp_path / "s1"
    s1.mkdir(exist_ok=True)
    tsv = TSV.replace(b"\t256\t", f"\t{budget}\t".encode())
    (s1 / f"protection_b{budget}.conditions.tsv").write_bytes(tsv)
    (s1 / f"protection_b{budget}.json").write_text(
        json.dumps(
            {
                "status": "FINAL",
                "sufficient_statistics_sha256": hashlib.sha256(tsv).hexdigest(),
                "dependencies": {"sha256": fingerprint},
                "inputs": {
                    "grid_pp": f"grid-pp{budget}",
                    "grid_base": f"grid-base{budget}",
                },
                "design": {"presses": ["random_pp"]},
            }
        )
    )
    return s1


def _current(runs, budget):
    sys.path.insert(0, str(ROOT / "scripts"))
    import report_s

    return report_s.protection_dependencies(runs, budget)["sha256"]


def test_final_current_evidence_publishes_and_stale_evidence_refuses(tmp_path):
    runs = tmp_path / "runs"
    for b in (256, 1024):
        _grid(runs, f"grid-pp{b}", [_row("a", 984, budget=b), _row("b", 56, budget=b)])
        _grid(
            runs,
            f"grid-base{b}",
            [_row("a", 984, "random", b), _row("b", 56, "random", b)],
        )
        s1 = _s1(tmp_path, runs, b, _current(runs, b))
    rc, out = _cli(runs, s1, tmp_path / "rc.json")
    assert (rc, out["status"]) == (0, "FINAL"), out
    assert out["cells"]["b256"]["random_pp"]["n_budget_relaxed"] == 1
    assert out["cells"]["b1024"]["random_pp"]["n_ratio_one"] == 2
    # a row changes after the FINAL S.1 read -> the evidence is stale
    _grid(runs, "grid-pp256", [_row("a", 900), _row("b", 56)])
    rc, out = _cli(runs, s1, tmp_path / "rc.json")
    assert (rc, out["status"]) == (2, "INVALID") and "cells" not in out
    assert "stale" in json.dumps(out["evidence"]["b256"]["problems"])

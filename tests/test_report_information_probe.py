"""CLI acceptance for the information-probe reader (Meeting 5, M5-C): analytic
channels written as rows, read through the real script, give the hand-known
values; incomplete and malformed reads carry no label."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from kvcache import task_information as ti

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_information_probe.py"
DESIGN = {
    "policies": ["full", "random"],
    "reference_policy": "full",
    "contexts": ["c0", "c1", "c2"],
    "eviction_seeds": [0, 1],
    "queries": ["s", "n"],
}
KIND = {"full": "identity", "random": "uniform"}


def _rows():
    out = []
    for pol in DESIGN["policies"]:
        for q in DESIGN["queries"]:
            for c in DESIGN["contexts"]:
                for e in DESIGN["eviction_seeds"]:
                    for r in ti.reference_channel(KIND[pol], query=q):
                        out.append(
                            {
                                "policy": pol,
                                "query": q,
                                "context_id": c,
                                "eviction_seed": e,
                                "s": r.s,
                                "n": r.n,
                                "symbol_logprobs": list(r.logprobs),
                                "greedy": r.greedy,
                            }
                        )
    return out


def _run(tmp_path, rows, design=DESIGN):
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "design.json").write_text(json.dumps(design))
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rows",
            str(tmp_path / "rows.jsonl"),
            "--design",
            str(tmp_path / "design.json"),
            "--out",
            str(tmp_path / "read.json"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p, json.loads((tmp_path / "read.json").read_text())


def test_final_read_reproduces_the_analytic_values(tmp_path):
    p, read = _run(tmp_path, _rows())
    assert p.returncode == 0, p.stderr
    assert read["status"] == "FINAL"
    assert read["n_complete_cells"] == read["n_expected_cells"] == 24
    assert read["not"] == "not total cache MI"
    full, rnd = read["policies"]["full"], read["policies"]["random"]
    for q in ("query_s", "query_n"):
        assert full[q]["mi_bits"]["mean"] == pytest.approx(2.0)
        assert full[q]["expected_correct_prob"]["mean"] == pytest.approx(1.0)
        assert rnd[q]["mi_bits"]["mean"] == pytest.approx(0.0, abs=1e-12)
        assert rnd[q]["expected_correct_prob"]["mean"] == pytest.approx(0.25)
        assert rnd[q]["log_loss_bits"]["mean"] == pytest.approx(2.0)
        assert rnd[q]["mi_bits_minus_full"]["mean"] == pytest.approx(-2.0)
        assert rnd[q]["mi_bits_minus_full"]["ci95"] == pytest.approx([-2.0, -2.0])
    assert read["bootstrap"] == {"unit": "context", "n_resamples": 2000, "seed": 0}


def test_missing_variant_is_partial_with_no_numbers(tmp_path):
    rows = _rows()[1:]
    p, read = _run(tmp_path, rows)
    assert p.returncode == 0
    assert read["status"] == "PARTIAL" and "policies" not in read
    assert read["n_complete_cells"] == 23


def test_duplicate_or_foreign_rows_are_invalid(tmp_path):
    rows = _rows()
    p, read = _run(tmp_path, rows + rows[:1])
    assert p.returncode == 2 and read["status"] == "INVALID"
    assert "policies" not in read
    foreign = dict(rows[0], policy="h2o")
    p, read = _run(tmp_path, rows + [foreign])
    assert read["status"] == "INVALID" and "not in the design" in read["problems"][0]
    bad = dict(rows[0], symbol_logprobs=[0.5, -1, -1, -1])
    p, read = _run(tmp_path, [bad] + rows[1:])
    assert read["status"] == "INVALID"


def test_non_finite_metric_is_reported_without_a_number(tmp_path):
    rows = [
        (
            dict(r, symbol_logprobs=[-math.inf] * 4, greedy=None)
            if r["policy"] == "random"
            else r
        )
        for r in _rows()
    ]
    p, read = _run(tmp_path, rows)
    assert read["status"] == "FINAL"
    ll = read["policies"]["random"]["query_s"]["log_loss_bits"]
    assert ll == {"mean": None, "ci95": None, "non_finite": True}
    assert read["policies"]["random"]["query_s"]["other_prob"]["mean"] == 1.0


def test_self_check_prints_the_reference_table():
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    rows = {r["channel"]: r for r in json.loads(p.stdout)}
    assert rows["permutation"]["mi_bits"] == pytest.approx(2.0)
    assert rows["permutation"]["expected_correct_prob"] == 0.0

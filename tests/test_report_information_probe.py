"""CLI acceptance for the information-probe reader (Meeting 5, M5-C; review of
6b409001, R3): analytic channels in reference mode give the hand-known values;
production rows need the versioned schema, the design's provenance and passed
controls; fractional keys, an impossible greedy symbol, failed controls, wrong
provenance and torn files are INVALID (exit 2); missing variants PARTIAL
(exit 3); a failed read retires the old FINAL; the registered paired contrast
is computed from paired context values."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from kvcache import task_information as ti

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_information_probe.py"
KIND = {"full": "identity", "random": "uniform", "h2o_style": "permutation"}
PROV = {
    "implementation_id": "i" * 64,
    "model_revision": "m",
    "tokenizer_revision": "m",
    "answer_alphabet_sha256": "a" * 64,
    "task_layout_sha256": "t" * 64,
    "design_sha256": "d" * 64,
}


def _design(mode="reference"):
    d = {
        "mode": mode,
        "policies": ["full", "random", "h2o_style"],
        "reference_policy": "full",
        "contexts": ["c0", "c1", "c2"],
        "eviction_seeds": [0, 1],
        "queries": ["s", "n"],
        "paired_contrasts": [["random", "h2o_style"]],
    }
    if mode == "production":
        d["expected_provenance"] = PROV
    return d


def _rows(mode="reference"):
    out = []
    d = _design(mode)
    for pol in d["policies"]:
        for q in d["queries"]:
            for c in d["contexts"]:
                for e in d["eviction_seeds"]:
                    for r in ti.reference_channel(KIND[pol], query=q):
                        row = {
                            "policy": pol,
                            "query": q,
                            "context_id": c,
                            "eviction_seed": e,
                            "s": r.s,
                            "n": r.n,
                            "symbol_logprobs": list(r.logprobs),
                            "greedy": r.greedy,
                        }
                        if mode == "production":
                            row.update(
                                schema=ti.ROW_SCHEMA,
                                provenance=dict(PROV),
                                controls={c2: True for c2 in ti.REQUIRED_CONTROLS},
                                top1_logprob=max(
                                    x for x in r.logprobs if x != -math.inf
                                ),
                            )
                        out.append(row)
    return out


def _run(tmp_path, rows, design, extra=()):
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
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads((tmp_path / "read.json").read_text())


@pytest.mark.parametrize("mode", ["reference", "production"])
def test_final_read_reproduces_the_analytic_values(tmp_path, mode):
    rc, read = _run(tmp_path, _rows(mode), _design(mode))
    assert (rc, read["status"]) == (0, "FINAL"), read.get("problems")
    assert read["not"] == "not total cache MI" and read["mode"] == mode
    full, rnd = read["policies"]["full"], read["policies"]["random"]
    for q in ("query_s", "query_n"):
        assert full[q]["mi_bits"]["mean"] == pytest.approx(2.0)
        assert rnd[q]["mi_bits"]["mean"] == pytest.approx(0.0, abs=1e-12)
        assert rnd[q]["expected_correct_prob"]["mean"] == pytest.approx(0.25)
        assert rnd[q]["mi_bits_minus_full"]["mean"] == pytest.approx(-2.0)
    # registered paired contrast: random (0 bits) - h2o_style (wrong permutation,
    # 2 bits) = -2, from paired context values
    pc = read["paired_contrasts"]["random-h2o_style"]["query_s"]
    assert pc["mean"] == pytest.approx(-2.0) and pc["ci95"] == pytest.approx([-2, -2])


def _mut(mode, fn):
    rows = copy.deepcopy(_rows(mode))
    fn(rows)
    return rows


INVALID_CASES = {
    # review of 6b409001, R3 (each was FINAL, exit 0)
    "fractional_keys": ("reference", lambda r: r[0].update(s=0.7, eviction_seed=0.9)),
    "impossible_greedy": ("reference", lambda r: r[0].update(greedy=1)),
    "failed_controls": (
        "production",
        lambda r: r[0]["controls"].update(clone_agreement=False),
    ),
    "wrong_provenance": (
        "production",
        lambda r: r[0]["provenance"].update(implementation_id="wrong"),
    ),
    # additional contract checks
    "no_schema": ("production", lambda r: r[0].pop("schema")),
    "bool_key": ("reference", lambda r: r[0].update(n=True)),
    "fixture_with_provenance": (
        "reference",
        lambda r: r[0].update(provenance={"implementation_id": "x"}),
    ),
    "duplicate": ("reference", lambda r: r.append(copy.deepcopy(r[0]))),
    "foreign_cell": ("reference", lambda r: r.append({**r[0], "policy": "h2o"})),
    "invalid_probability": (
        "reference",
        lambda r: r[0].update(symbol_logprobs=[0.5, -1, -1, -1]),
    ),
    "top1_mismatch": ("production", lambda r: r[0].update(top1_logprob=-3.0)),
}


@pytest.mark.parametrize("name", sorted(INVALID_CASES))
def test_invalid_rows_through_the_real_cli(tmp_path, name):
    mode, fn = INVALID_CASES[name]
    rc, read = _run(tmp_path, _mut(mode, fn), _design(mode))
    assert (rc, read["status"]) == (2, "INVALID"), read.get("problems")
    assert "policies" not in read


def test_missing_variant_is_partial_exit_3_without_numbers(tmp_path):
    rc, read = _run(tmp_path, _rows()[1:], _design())
    assert (rc, read["status"]) == (3, "PARTIAL") and "policies" not in read
    rc, read = _run(tmp_path, _rows()[1:], _design(), extra=["--partial"])
    assert (rc, read["status"]) == (0, "PARTIAL")


def test_torn_file_retires_the_old_final(tmp_path):
    rc, read = _run(tmp_path, _rows(), _design())
    assert read["status"] == "FINAL"
    path = tmp_path / "rows.jsonl"
    path.write_text(path.read_text() + '\n{"torn":')
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--rows",
            str(path),
            "--design",
            str(tmp_path / "design.json"),
            "--out",
            str(tmp_path / "read.json"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    read = json.loads((tmp_path / "read.json").read_text())
    assert (p.returncode, read["status"]) == (2, "INVALID")
    assert read["retired_previous_output"].startswith("read.json.superseded")


def test_production_design_must_declare_provenance(tmp_path):
    d = _design("production")
    d.pop("expected_provenance")
    rc, read = _run(tmp_path, _rows("production"), d)
    assert (rc, read["status"]) == (2, "INVALID")


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

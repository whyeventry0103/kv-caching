"""M5-A protection-by-policy read (review of 6b409001, R1): the REAL CLI on
schema-complete fixture runs. A positive control reads FINAL with hand-known
estimates; every injected defect from the review is INVALID (exit 2) or, for a
missing condition, PARTIAL (exit 3); a failed read never leaves an old FINAL
standing. Plus the TOST / equivalence-family rules."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from kvcache.metrics import policy_replication as pr
from kvcache.metrics.protocol_audit import NOT_TESTABLE

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_policy_replication.py"
SPEC = yaml.safe_load((ROOT / "configs/meeting5/m5a_read.yaml").read_text())
MODEL = SPEC["model"]
GRID100 = json.loads((ROOT / "configs/meeting3/grid100_pids.json").read_text())["pids"]
STRICT = json.loads((ROOT / SPEC["cohort_source"]).read_text())["pids"]
MANIFEST = json.loads((ROOT / "configs/meeting5/input_manifest_m5.json").read_text())
# accuracy pattern per press: correct iff the problem index is below the cut
CUT = {
    "random": 60,
    "random_pp": 80,
    "streamingllm": 50,
    "streamingllm_pp": 80,
    "h2o": 20,
    "h2o_style_pp_v1": 70,
    "rkv": 40,
    "rkv_pp_v1": 75,
}


def _source_rows(run: str) -> list[dict]:
    from kvcache.m5_runs import expected_identity

    src = SPEC["sources"][run]
    cfg = yaml.safe_load((ROOT / src["config"]).read_text())
    ident = expected_identity(cfg) if src["provenance"] == "m5" else None
    rows = []
    for press in cfg["presses"]:
        for seed in cfg["seeds"]:
            for i, pid in enumerate(GRID100):
                r = {
                    "pid": pid,
                    "press": press,
                    "budget": 256,
                    "model": MODEL,
                    "bench": "math500",
                    "decoding": {"seed": seed},
                    "trace": "t",
                    "correct": i < CUT[press],
                    "truncated": False,
                    "runmeta": {"commit": "abc", "source_tree_hash": "x"},
                    "versions": {},
                }
                if ident:
                    r["provenance"] = {
                        **ident,
                        "inputs_sha256": MANIFEST["prompts"][pid],
                    }
                rows.append(r)
    return rows


ALL = {run: _source_rows(run) for run in SPEC["sources"]}


def _write(runs: Path, data: dict[str, list[dict]], torn: str | None = None) -> None:
    for run, rows in data.items():
        d = runs / run / "math500" / MODEL
        d.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(r) for r in rows) + "\n"
        if run == torn:
            text += '{"torn":'
        (d / "rows.jsonl").write_text(text)


def _cli(tmp_path, data, *, cohort="strict", torn=None, fresh=True):
    runs, out = tmp_path / "runs", tmp_path / "out"
    if fresh and runs.exists():
        import shutil

        shutil.rmtree(runs)
    _write(runs, data, torn)
    p = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cohort",
            cohort,
            "--runs-root",
            str(runs),
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    d = json.loads((out / f"m5a_read_{cohort}.json").read_text())
    return p.returncode, d


def test_positive_control_is_final_with_hand_known_estimates(tmp_path):
    rc, d = _cli(tmp_path, ALL)
    assert (rc, d["status"]) == (0, "FINAL"), d.get("sources")
    est = d["estimates"]
    # strict cohort = GRID100 minus 3 relaxed problems; the cut is by GRID100
    # index, so recompute the expected accuracy over the cohort itself
    idx = {p: i for i, p in enumerate(GRID100)}

    def acc(press):
        return sum(idx[p] < CUT[press] for p in STRICT) / len(STRICT)

    assert est["accuracy"]["h2o+P"]["mean"] == pytest.approx(acc("h2o_style_pp_v1"))
    h = est["contrasts"]["random-h2o"]
    want = (acc("random_pp") - acc("h2o_style_pp_v1")) - (acc("random") - acc("h2o"))
    assert h["interaction"]["mean"] == pytest.approx(want)
    assert est["registered_inference"]["family_size"] == 2
    eq = est["equivalence_protected_contrasts"]
    assert eq["primary"]["margin"] == 0.03 and eq["sensitivity"]["margin"] == 0.05
    # identical protected Random and StreamingLLM arms: zero SE -> NOT TESTABLE
    s = eq["primary"]["members"]["random-streamingllm"]
    assert s["label"] == NOT_TESTABLE and s["observed_mean"] == 0.0
    assert d["dependencies"]["sha256"] and d["capacity_matched"] is True


def _mut(run, fn):
    data = copy.deepcopy(ALL)
    fn(data[run])
    return data


CASES = {
    "string_False": (
        _mut("grid-pp256", lambda rs: [r.update(correct="False") for r in rs]),
        "INVALID",
    ),
    "nan": (_mut("grid-pp256", lambda rs: rs[0].update(correct=math.nan)), "INVALID"),
    "foreign_seed": (
        _mut(
            "grid-h2opp256-m5",
            lambda rs: rs.append({**copy.deepcopy(rs[0]), "decoding": {"seed": 999}}),
        ),
        "INVALID",
    ),
    "duplicate": (
        _mut("grid-h2opp256-m5", lambda rs: rs.append(copy.deepcopy(rs[0]))),
        "INVALID",
    ),
    "wrong_model": (
        _mut("grid-diss256-h2o", lambda rs: rs[0].update(model="wrong_model")),
        "INVALID",
    ),
    "identity_on_legacy_row": (
        _mut(
            "grid-diss256-h2o",
            lambda rs: rs[0].update(provenance={"implementation_id": "wrong"}),
        ),
        "INVALID",
    ),
    "wrong_m5_identity": (
        _mut(
            "grid-rkvpp256-m5",
            lambda rs: rs[0]["provenance"].update(implementation_id="0" * 64),
        ),
        "INVALID",
    ),
    "wrong_prompt_hash": (
        _mut(
            "grid-rkvpp256-m5",
            lambda rs: rs[0]["provenance"].update(inputs_sha256="0" * 64),
        ),
        "INVALID",
    ),
    "truncated_scored_correct": (
        _mut("grid-pp256", lambda rs: rs[0].update(truncated=True, correct=True)),
        "INVALID",
    ),
    "foreign_outside_cohort": (
        # a relaxed problem is outside the strict cohort but inside the source
        # design: its duplicate is still INVALID (validated before selection)
        _mut(
            "grid-diss256-tok",
            lambda rs: rs.append(
                copy.deepcopy(next(r for r in rs if r["pid"] not in STRICT))
            ),
        ),
        "INVALID",
    ),
    "missing_row": (_mut("grid-pp256", lambda rs: rs.pop()), "PARTIAL"),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_review_defects_are_refused_through_the_real_cli(tmp_path, name):
    data, want = CASES[name]
    rc, d = _cli(tmp_path, data)
    assert d["status"] == want, (name, d.get("sources"))
    assert rc == {"INVALID": 2, "PARTIAL": 3}[want]
    assert "estimates" not in d


def test_absent_source_is_partial_and_whole_cohort_needs_capmatch(tmp_path):
    data = {k: v for k, v in ALL.items() if k != "grid-rkvpp256-m5"}
    rc, d = _cli(tmp_path, data)
    assert (rc, d["status"]) == (3, "PARTIAL")
    assert d["sources"]["grid-rkvpp256-m5"] == {"status": "PARTIAL", "absent": True}
    rc, d = _cli(tmp_path, ALL, cohort="whole")
    assert (rc, d["status"]) == (3, "PARTIAL") and d["capacity_matched"] is False


def test_torn_json_retires_the_old_final_and_writes_invalid(tmp_path):
    rc, d = _cli(tmp_path, ALL)
    assert d["status"] == "FINAL"
    rc, d = _cli(tmp_path, ALL, torn="grid-diss256-h2o")
    assert (rc, d["status"]) == (2, "INVALID") and "malformed JSON" in d["error"]
    assert d["retired_previous_output"].startswith("m5a_read_strict.json.superseded")
    retired = sorted(p.name for p in (tmp_path / "out").glob("*.superseded-*"))
    # the old FINAL and its condition table are both no longer current
    assert [n.split(".superseded-")[0] for n in retired] == [
        "m5a_read_strict.conditions.tsv",
        "m5a_read_strict.json",
    ]
    assert not (tmp_path / "out" / "m5a_read_strict.conditions.tsv").exists()


def test_tost_rules():
    within = pr.tost([0.01, -0.01, 0.0, 0.005, -0.005] * 40, margin=0.03, alpha=0.05)
    assert within["label"] == "within margin" and within["p_tost"] < 0.05
    wide = pr.tost([1.0, -1.0, 0.5, -0.5] * 5, margin=0.03, alpha=0.05)
    assert wide["label"] == "not shown within margin"
    flat = pr.tost([0.2] * 10, margin=0.03, alpha=0.05)
    assert flat["label"] == NOT_TESTABLE and flat["observed_mean"] == pytest.approx(0.2)
    fam = pr.equivalence_family(
        {"a": [0.0, 0.01, -0.01] * 50, "b": [1.0, -1.0] * 5}, margin=0.03, alpha=0.05
    )
    assert fam["members"]["a"]["equivalent_per_comparison"] is True
    assert fam["joint_within_margin"] is False  # IUT: b is not within margin


def test_final_read_publishes_a_condition_table_that_reproduces_it(tmp_path):
    """Review of 94720ff: the read publishes its sufficient statistics (one row
    per (problem, seed): every arm's correctness and truncation flag), bound to
    the read by sha256, and the table alone reproduces every estimate."""
    import csv
    import hashlib

    rc, d = _cli(tmp_path, ALL)
    assert (rc, d["status"]) == (0, "FINAL")
    tsv = tmp_path / "out" / d["sufficient_statistics_tsv"]
    assert hashlib.sha256(tsv.read_bytes()).hexdigest() == (
        d["sufficient_statistics_sha256"]
    )
    with tsv.open() as fh:
        r = list(csv.reader(fh, delimiter="\t"))
    header, rows = r[0], r[1:]
    seeds = SPEC["seeds"]
    assert len(rows) == len(STRICT) * len(seeds) and header[:2] == ["pid", "seed"]
    assert len(header) == 2 + 2 * 2 * len(SPEC["arms"])  # 8 arms x (correct, trunc)
    values = pr.values_from_table(header, rows)
    again = pr.replication_read(
        values,
        expected={(p, s) for p in STRICT for s in seeds},
        reference=SPEC["reference"],
        pairings=SPEC["pairings"],
        descriptive=SPEC["descriptive"],
        equivalence=SPEC["equivalence"],
    )
    assert json.loads(json.dumps(again)) == d["estimates"]
    # a non-FINAL read publishes no table (and retires an old one)
    rc, d = _cli(tmp_path, {k: v for k, v in ALL.items() if k != "grid-pp256"})
    assert d["status"] == "PARTIAL" and "sufficient_statistics_tsv" not in d
    assert not tsv.exists()

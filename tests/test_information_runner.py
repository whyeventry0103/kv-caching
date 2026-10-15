"""Information-calibration runner on CPU (review of d50aa40, item 4): the REAL
``execute`` path of scripts/run_information_probe.py with stubs only for the
model, tokenizer, filler and presses (tests/test_information_probe.py), then
the REAL reader CLI with the frozen calibration criteria. A working channel
reads FINAL and PASSES; a channel whose eviction never reaches the facts FAILS
on the frozen criteria; resume skips finished blocks and refuses a changed
design; the held flag and the smoke gates refuse real execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_information_probe as rip  # noqa: E402
import test_information_probe as tip  # noqa: E402

import kvcache.models as km  # noqa: E402
from kvcache import information_calibration as ic  # noqa: E402
from kvcache import information_probe as ip  # noqa: E402

REAL = yaml.safe_load((ROOT / "configs/meeting5/info_calibration.yaml").read_text())


def _cfg(tmp_path, budget=40):
    cfg = {
        "name": "info_calibration_test",
        "held": False,
        "models": ["r1_distill_qwen_7b"],
        "design_seed": 3,
        "n_blocks": 2,
        "values": tip.VALUES,
        "aliases": tip.ALIASES,
        "spec": {"body_tokens": 120, "early_offset": 10, "late_offset": 90},
        "policies": ["full", "recency"],
        "budget": budget,
        "compression_interval": 8,
        "eviction_seeds": [0],
        "queries": ["s", "n"],
        "paired_contrasts": [],
        "calibration_criteria": REAL["calibration_criteria"],
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def _stubs(monkeypatch, budget):
    backend = tip.PressedStub()
    tok = SimpleNamespace(encode=lambda t, add_special_tokens=False: tip.encode(t))

    def policies(model, cfg):
        def full(seed):
            backend.press = None
            return ip.NoPress()

        def recency(seed):
            p = tip.Recency(None, 0, budget)
            backend.press = p
            return p

        return {"full": full, "recency": recency}

    monkeypatch.setattr(rip, "load_passages", lambda cfg, base: tip.PASSAGES)
    monkeypatch.setattr(rip, "smoke_gates", lambda cfg, base: [])
    monkeypatch.setattr(rip, "make_backend", lambda model: backend)
    monkeypatch.setattr(rip, "policy_factories", policies)
    monkeypatch.setattr(km, "load_tokenizer", lambda cfg: tok)
    monkeypatch.setattr(
        km, "load_model_and_tokenizer", lambda cfg, **k: (object(), tok)
    )


def _read(tmp_path, out, cfg_path):
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/report_information_probe.py"),
            "--rows",
            str(out / "readout"),
            "--design",
            str(out / "design.json"),
            "--out",
            str(tmp_path / "read.json"),
            "--calibration-criteria",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads((tmp_path / "read.json").read_text())


def test_working_channel_reads_final_and_passes(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    res = rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert res == {"rows_written": 2 * 2 * 16 * 2, "blocks": 2, "skipped": 0}
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (0, "FINAL"), read.get("problems")
    cal = read["calibration"]
    assert cal["verdict"] == "PASS", cal
    assert cal["criteria"]["C4_end_to_end"]["value"] == 0.0
    assert cal["criteria"]["C5_eviction_reaches_facts"]["value"] == 0.0
    assert cal["criteria"]["C6_dynamic_range"]["value"] == pytest.approx(1.0, abs=1e-3)


def test_no_eviction_fails_the_frozen_criteria(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path, budget=10_000)
    _stubs(monkeypatch, budget=10_000)  # the "policy" never evicts anything
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    rc, read = _read(tmp_path, out, path)
    assert read["status"] == "FINAL"
    assert read["calibration"]["verdict"] == "FAIL"
    assert read["calibration"]["failed"] == [
        "C5_eviction_reaches_facts",
        "C6_dynamic_range",
    ]


def test_resume_skips_finished_blocks_and_refuses_a_changed_design(
    tmp_path, monkeypatch
):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    again = rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert again["rows_written"] == 0 and again["skipped"] == 2
    changed = {**cfg, "design_seed": 4}
    with pytest.raises(SystemExit, match="design.json differs"):
        rip.execute(changed, out_dir=out, base_runs=tmp_path)


def test_real_execution_refuses_while_held_or_without_smokes(tmp_path, monkeypatch):
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_information_probe.py"),
            "--i-understand",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"KVCACHE_REMOTE": "1", "PATH": "/usr/bin:/bin"},
    )
    assert p.returncode != 0 and "HELD" in p.stderr
    cfg, _ = _cfg(tmp_path)
    monkeypatch.setattr(rip, "smoke_gates", lambda cfg, base: ["no PASS marker"])
    with pytest.raises(SystemExit, match="smoke gate"):
        rip.execute(cfg, out_dir=tmp_path / "run", base_runs=tmp_path)


def test_evaluator_refuses_a_non_final_read():
    out = ic.evaluate({"status": "PARTIAL"}, [], REAL["calibration_criteria"])
    assert out["verdict"] == "FAIL" and not out["criteria"]["C3_read_final"]["passed"]


# --- review of 5616630: every counterexample now refuses (R1-R3) ----------------


def _rewrite(out, mutate):
    """Apply ``mutate`` to all stored rows (as the review's repro does)."""
    files = sorted((out / "readout").glob("*.jsonl"))
    rows = [json.loads(x) for p in files for x in p.read_text().splitlines()]
    mutate(rows)
    for p in files:
        p.unlink()
    (out / "readout" / "all.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )


def _missing_63_of_64_e2e(rows):
    present = [r for r in rows if r["e2e_max_abs_diff"] is not None]
    assert len(present) == 64
    for r in present[1:]:
        r["e2e_max_abs_diff"] = None


def _nan_e2e(rows):
    [r for r in rows if r["e2e_max_abs_diff"] is not None][-1]["e2e_max_abs_diff"] = (
        float("nan")
    )


def _negative_retention(rows):
    for r in rows:
        if r["policy"] != "full":
            r["retained_fraction"] = {"s": -1.0, "n": -1.0}


def _e2e_on_unscheduled_query(rows):
    next(r for r in rows if r["query"] == "s")["e2e_max_abs_diff"] = 0.0


def _missing_retention(rows):
    next(r for r in rows if r["policy"] != "full")["retained_fraction"]["s"] = None


def _bool_prunes(rows):
    rows[0]["n_prunes"] = True


@pytest.mark.parametrize(
    "mutate, why",
    [
        (_missing_63_of_64_e2e, "end-to-end control missing"),
        (_nan_e2e, "end-to-end control missing or invalid: nan"),
        (_negative_retention, "outside [0, 1]"),
        (_e2e_on_unscheduled_query, "query the schedule does not measure"),
        (_missing_retention, "missing or outside [0, 1]"),
        (_bool_prunes, "n_prunes is not a non-negative integer"),
    ],
)
def test_malformed_calibration_measurements_make_the_read_invalid(
    tmp_path, monkeypatch, mutate, why
):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    _rewrite(out, mutate)
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (2, "INVALID")
    assert any(why in p for p in read["problems"]), read["problems"]
    assert "calibration" not in read or read["calibration"]["verdict"] == "FAIL"


def test_all_compressed_at_floor_reads_final_but_fails_c6(tmp_path, monkeypatch):
    """Four compressed arms that all lose both facts: valid data, a FAILED
    calibration (the review's R3 case PASSED before the floor rule)."""
    cfg, path = _cfg(tmp_path, budget=5)
    cfg["policies"] = ["full", "rec_a", "rec_b", "rec_c", "rec_d"]
    path.write_text(yaml.safe_dump(cfg))
    _stubs(monkeypatch, budget=5)
    base = rip.policy_factories

    def four(model, c):
        b = base(model, c)
        return {"full": b["full"], **{p: b["recency"] for p in c["policies"][1:]}}

    monkeypatch.setattr(rip, "policy_factories", four)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (0, "FINAL")
    cal = read["calibration"]
    assert cal["verdict"] == "FAIL" and cal["failed"] == ["C6_dynamic_range"]
    r = cal["criteria"]["C6_dynamic_range"]["ranges_bits"]
    assert r["full_to_compressed"] == pytest.approx(2.0, abs=1e-3)
    assert r["compressed_only"] == pytest.approx(0.0, abs=1e-3)
    assert r["max_compressed"] < REAL["calibration_criteria"]["floor_mi_bits"]


def test_equal_informative_compressed_arms_are_a_valid_null(tmp_path, monkeypatch):
    """C6 never requires a policy order: two identical compressed arms that
    each keep the late fact (1 bit) pass calibration."""
    cfg, path = _cfg(tmp_path)
    cfg["policies"] = ["full", "rec_a", "rec_b"]
    path.write_text(yaml.safe_dump(cfg))
    _stubs(monkeypatch, budget=40)
    base = rip.policy_factories
    monkeypatch.setattr(
        rip,
        "policy_factories",
        lambda m, c: {
            "full": base(m, c)["full"],
            "rec_a": base(m, c)["recency"],
            "rec_b": base(m, c)["recency"],
        },
    )
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    rc, read = _read(tmp_path, out, path)
    cal = read["calibration"]
    assert (rc, cal["verdict"]) == (0, "PASS"), cal
    assert cal["criteria"]["C6_dynamic_range"]["ranges_bits"][
        "compressed_only"
    ] == pytest.approx(0.0, abs=1e-3)


def test_a_substituted_criteria_file_is_refused(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    loose = {
        **cfg,
        "calibration_criteria": {
            **cfg["calibration_criteria"],
            "max_mean_early_span_retained": 1.0,
        },
    }
    other = tmp_path / "loose.yaml"
    other.write_text(yaml.safe_dump(loose))
    rc, read = _read(tmp_path, out, other)
    assert (rc, read["status"]) == (2, "INVALID")
    assert "differs from the criteria frozen in the design" in read["error"]


def test_tampered_or_missing_input_manifest_is_invalid(tmp_path, monkeypatch):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    m = out / rip.MANIFEST_NAME
    man = json.loads(m.read_text())
    man["passages"][0]["n_tokens"] += 1
    m.write_text(json.dumps(man))
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (2, "INVALID")
    m.unlink()
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["status"]) == (2, "INVALID")


def test_require_calibration_pass_separates_validity_from_acceptance(
    tmp_path, monkeypatch
):
    for budget, want in ((40, 0), (10_000, rip_report_not_accepted())):
        root = tmp_path / str(budget)
        root.mkdir()
        cfg, path = _cfg(root, budget=budget)
        _stubs(monkeypatch, budget=budget)
        rip.execute(cfg, out_dir=root / "run", base_runs=root)
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_information_probe.py"),
                "--rows",
                str(root / "run" / "readout"),
                "--design",
                str(root / "run" / "design.json"),
                "--out",
                str(root / "read.json"),
                "--require-calibration-pass",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        read = json.loads((root / "read.json").read_text())
        assert p.returncode == want and read["status"] == "FINAL"
        assert ic.accepted(read) is (want == 0)


def rip_report_not_accepted():
    sys.path.insert(0, str(ROOT / "scripts"))
    import report_information_probe as rep

    return rep.NOT_ACCEPTED


# --- review of 5616630, R2: the design binds the ACTUAL selected inputs ----------


def _no_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model weights were loaded")

    monkeypatch.setattr(km, "load_model_and_tokenizer", boom)


def _changed_filler():
    return [
        it_mod().Passage("changed-" + p.source, tuple(reversed(p.token_ids)))
        for p in tip.PASSAGES
    ]


def it_mod():
    from kvcache import information_tasks

    return information_tasks


def _snapshot(out):
    return {p: p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_unchanged_inputs_resume_skips_finished_blocks(tmp_path, monkeypatch):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    before = _snapshot(out)
    assert rip.execute(cfg, out_dir=out, base_runs=tmp_path) == {
        "rows_written": 0,
        "blocks": 2,
        "skipped": 2,
    }
    assert _snapshot(out) == before


@pytest.mark.parametrize("partial", [False, True])
def test_changed_filler_resume_refuses_before_model_load(
    tmp_path, monkeypatch, partial
):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    if partial:  # one block finished, the other not yet written
        sorted((out / "readout").glob("*.jsonl"))[-1].unlink()
    before = _snapshot(out)
    monkeypatch.setattr(rip, "load_passages", lambda c, b: _changed_filler())
    _no_model(monkeypatch)
    with pytest.raises(SystemExit, match="REFUSED: an existing design.json differs"):
        rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert _snapshot(out) == before  # no design, manifest or row was rewritten


def test_partial_run_with_unchanged_inputs_completes_only_the_missing_block(
    tmp_path, monkeypatch
):
    cfg, path = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    sorted((out / "readout").glob("*.jsonl"))[-1].unlink()
    res = rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    assert res == {"rows_written": 2 * 16 * 2, "blocks": 2, "skipped": 1}
    rc, read = _read(tmp_path, out, path)
    assert (rc, read["calibration"]["verdict"]) == (0, "PASS")


def test_incomplete_stored_block_refuses_the_resume(tmp_path, monkeypatch):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    f = sorted((out / "readout").glob("*.jsonl"))[0]
    f.write_text("".join(x + "\n" for x in f.read_text().splitlines()[:-1]))
    _no_model(monkeypatch)
    with pytest.raises(SystemExit, match="incomplete or duplicated"):
        rip.execute(cfg, out_dir=out, base_runs=tmp_path)


def test_stored_rows_without_their_manifest_refuse(tmp_path, monkeypatch):
    cfg, _ = _cfg(tmp_path)
    _stubs(monkeypatch, budget=40)
    out = tmp_path / "run"
    rip.execute(cfg, out_dir=out, base_runs=tmp_path)
    (out / rip.MANIFEST_NAME).unlink()
    _no_model(monkeypatch)
    with pytest.raises(SystemExit, match="without their design.json and input"):
        rip.execute(cfg, out_dir=out, base_runs=tmp_path)


def test_manifest_binds_filler_content_not_only_ids():
    """Same sources, different token content -> a different manifest hash."""
    it = it_mod()
    enc = tip.encode

    def man(passages):
        blocks = it.make_blocks(
            1,
            design_seed=3,
            aliases=tip.ALIASES,
            passages=passages,
            encode=enc,
            values=tip.VALUES,
            spec=tip.SPEC,
        )
        v = {
            b.block_id: it.all_variants(
                b, passages=passages, encode=enc, values=tip.VALUES, spec=tip.SPEC
            )
            for b in blocks
        }
        return it.input_manifest(blocks, v, passages=passages, symbol_ids=[1])

    same_ids = [
        it.Passage(p.source, tuple(reversed(p.token_ids))) for p in tip.PASSAGES
    ]
    a, b = man(tip.PASSAGES), man(same_ids)
    assert [x["source"] for x in a["passages"]] == [x["source"] for x in b["passages"]]
    assert ip.sha256_json(a) != ip.sha256_json(b)

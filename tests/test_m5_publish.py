"""Review of 946c76a, issue 2: the M5 runner publishes row + sidecar PAIRS
through an atomic journal and recovers every interruption without regenerating
the interrupted condition; pre-journal orphans are quarantined (preserved, not
discarded) and regenerated under control; conflicts refuse; only owned seed
shards are touched; progress counts complete pairs only. Real runner, CPU
stubs for the model and generation (tests/test_run_grid_m5.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import nightly_autorun as na  # noqa: E402
import test_run_grid_m5 as tm  # noqa: E402

from kvcache import m5_publish as mp  # noqa: E402


class Interrupt(Exception):
    pass


def _counted(monkeypatch):
    n = {"generations": 0}
    real = tm.m5.generate_compressed

    def gen(*a, **k):
        n["generations"] += 1
        return real(*a, **k)

    monkeypatch.setattr(tm.m5, "generate_compressed", gen)
    return n


def _pairs_ok(out, n):
    rows, sides = tm._rows(out), tm._rows(out, "token_ids")
    keys = [mp.row_key(r) for r in rows]
    assert len(rows) == len(sides) == n and len(set(keys)) == n
    assert {mp.row_key(s) for s in sides} == set(keys)
    by = {mp.row_key(s): s for s in sides}
    assert all(by[mp.row_key(r)]["provenance"] == r["provenance"] for r in rows)
    return rows


def _once(monkeypatch, obj, name, when):
    """Make obj.name raise Interrupt the first time ``when(*args)`` holds."""
    real, state = getattr(obj, name), {"done": False}

    def wrapped(*a, **k):
        if not state["done"] and when(*a, **k):
            state["done"] = True
            raise Interrupt(name)
        return real(*a, **k)

    monkeypatch.setattr(obj, name, wrapped)


@pytest.mark.parametrize("stage", ["before_publication", "between", "after_both"])
def test_interruption_recovers_without_regenerating(tmp_path, monkeypatch, stage):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch)
    n = _counted(monkeypatch)
    if stage == "before_publication":
        _once(monkeypatch, tm.m5, "publish", lambda *a, **k: True)
    elif stage == "between":
        _once(monkeypatch, mp, "_publish_one", lambda p, r: "token_ids" in p.parts)
    else:
        _once(
            monkeypatch,
            mp.os,
            "replace",
            lambda a, b, *r, **k: str(b).endswith(".published.json"),
        )
    with pytest.raises(Interrupt):
        tm._run(monkeypatch, cfg)
    assert n["generations"] == 1
    pending = [
        p for p in (out / mp.JOURNAL).rglob("*.json") if "published" not in p.name
    ]
    assert len(pending) == 1  # the pair is safe in its envelope
    first = json.loads(pending[0].read_text())["row"]
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    rows = _pairs_ok(out, 100)
    assert n["generations"] == 100  # the interrupted condition was NOT redone
    assert next(r for r in rows if r["pid"] == first["pid"]) == first
    # a second resume is a no-op
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    assert n["generations"] == 100
    _pairs_ok(out, 100)


def _legacy_orphan(out, which):
    """A pre-journal half pair: a row (or sidecar) with no partner, no envelope."""
    victim = tm._rows(out)[0]["pid"]
    sub = "token_ids" if which == "row" else "math500"  # remove the partner
    for p in (out / sub).rglob("*.jsonl"):
        keep = [x for x in p.read_text().splitlines() if json.loads(x)["pid"] != victim]
        p.write_text("".join(k + "\n" for k in keep))
    return victim


@pytest.mark.parametrize("which", ["row", "sidecar"])
def test_pre_journal_orphan_is_quarantined_then_regenerated(
    tmp_path, monkeypatch, which
):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch)
    assert tm._run(monkeypatch, cfg) == 0
    victim = _legacy_orphan(out, which)
    n = _counted(monkeypatch)
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    _pairs_ok(out, 100)
    assert n["generations"] == 1  # controlled regeneration of that condition only
    q = [
        json.loads(x)
        for p in (out / mp.QUARANTINE).rglob("*.jsonl")
        for x in p.read_text().splitlines()
    ]
    assert len(q) == 1 and json.loads(q[0]["line"])["pid"] == victim
    assert q[0]["reason"] == f"{which} without {'sidecar' if which == 'row' else 'row'}"


def test_review_case_row_without_sidecar_now_resumes(tmp_path, monkeypatch):
    """The exact reproduction: interrupted right after the row reached disk."""
    cfg, out, calls = tm._setup(tmp_path, monkeypatch)
    _once(monkeypatch, mp, "_publish_one", lambda p, r: "token_ids" in p.parts)
    with pytest.raises(Interrupt):
        tm._run(monkeypatch, cfg)
    assert (len(tm._rows(out)), len(tm._rows(out, "token_ids"))) == (1, 0)
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    _pairs_ok(out, 100)


def test_conflicting_evidence_refuses(tmp_path, monkeypatch):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch)
    _once(monkeypatch, mp, "_publish_one", lambda p, r: "token_ids" in p.parts)
    with pytest.raises(Interrupt):
        tm._run(monkeypatch, cfg)
    env_path = next((out / mp.JOURNAL).rglob("*.json"))
    env = json.loads(env_path.read_text())
    # (a) the envelope disagrees with the row already published -> conflict
    env["row"]["trace"] = "a different generation"
    env["pair_sha256"] = mp.hashlib.sha256(
        mp._canon([env["row"], env["sidecar"]]).encode()
    ).hexdigest()
    env_path.write_text(json.dumps(env))
    with pytest.raises(SystemExit, match="conflicting record"):
        tm._run(monkeypatch, cfg, "--resume")
    # (b) an envelope of another identity -> refused before publication
    env["row"]["provenance"]["implementation_id"] = "0" * 64
    env["sidecar"]["provenance"]["implementation_id"] = "0" * 64
    env["pair_sha256"] = mp.hashlib.sha256(
        mp._canon([env["row"], env["sidecar"]]).encode()
    ).hexdigest()
    env_path.write_text(json.dumps(env))
    with pytest.raises(SystemExit, match="identity differs"):
        tm._run(monkeypatch, cfg, "--resume")
    # (c) a tampered envelope fails its own hash
    env["row"]["trace"] = "tampered"
    env_path.write_text(json.dumps(env))
    with pytest.raises(SystemExit):
        tm._run(monkeypatch, cfg, "--resume")


def test_only_owned_shards_are_reconciled(tmp_path, monkeypatch):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch, cfg_over={"seeds": [11, 23]})
    _once(
        monkeypatch,
        mp,
        "_publish_one",
        lambda p, r: "token_ids" in p.parts and p.name.endswith("-s23.jsonl"),
    )
    with pytest.raises(Interrupt):
        tm._run(monkeypatch, cfg)  # seed 11 done, seed 23 interrupted
    s23_rows = next((out / "math500").rglob("*-s23.jsonl")).read_text()
    pending = [
        p for p in (out / mp.JOURNAL).rglob("*.json") if "published" not in p.name
    ]
    assert len(pending) == 1 and "-s23" in pending[0].parent.name
    # a process owning only seed 11 must leave seed 23's files alone
    assert tm._run(monkeypatch, cfg, "--resume", "--seeds", "11") == 0
    assert next((out / "math500").rglob("*-s23.jsonl")).read_text() == s23_rows
    assert pending[0].exists()
    assert tm._run(monkeypatch, cfg, "--resume", "--seeds", "23") == 0
    _pairs_ok(out, 200)


def test_scheduler_counts_complete_pairs_only(tmp_path, monkeypatch):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch)
    _once(monkeypatch, mp, "_publish_one", lambda p, r: "token_ids" in p.parts)
    with pytest.raises(Interrupt):
        tm._run(monkeypatch, cfg)
    job = na.Job(
        name="grid-rkvpp256-m5",
        command="grid_m5",
        config="c.yaml",
        run_name=out.name,
        total=1,
        seeds=[11],
        min_free_gb=28,
        progress_kind="grid_pairs",
        requires=None,
    )
    assert len(tm._rows(out)) == 1  # an orphan row exists ...
    assert na.count_progress(out.parent, job) == 0  # ... but completes nothing
    assert na.count_per_seed(out.parent, job) == {}
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    assert na.count_progress(out.parent, job) == 100
    assert na.count_per_seed(out.parent, job) == {11: 100}

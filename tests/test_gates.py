"""R.15 typed gates through the scheduler's PRODUCTION completion path.

Jobs come from the committed queue, expectations from committed configs; each
case goes through nightly_autorun.count_progress / launch_candidates, the code
the orchestrator runs, not through a detached helper.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "na_gates", ROOT / "scripts" / "nightly_autorun.py"
)
M = importlib.util.module_from_spec(SPEC)
sys.modules["na_gates"] = M
SPEC.loader.exec_module(M)

from kvcache.gates import _revision, _yaml  # noqa: E402
from kvcache.provenance import identity  # noqa: E402

_, JOBS = M.load_queue(ROOT / "configs" / "nightly_queue.yaml")
BY = {j.name: j for j in JOBS}


def _done(runs, name):
    return M.count_progress(runs, BY[name]) >= BY[name].total


def _dacc_identity():
    cfg = _yaml(ROOT / BY["dacc-gate-smoke"].config)
    return identity(
        "dacc", cfg, model_revision=_revision(cfg, ROOT), reference_run=None
    )


def _write_basis(runs, *, corrupt=False, wrong_identity=False):
    cfg = _yaml(ROOT / BY["q6-basis-fit"].config)
    ident = identity(
        "basis",
        cfg,
        model_revision=_revision(cfg, ROOT),
        reference_run=cfg["reference_run_name"],
    )
    if wrong_identity:
        ident = {**ident, "config_sha256": "old"}
    d = runs / "q6-basis-fit" / "basis"
    d.mkdir(parents=True)
    (d / "q6_basis.npz").write_bytes(b"basis-bytes")
    sha = hashlib.sha256(b"basis-bytes").hexdigest()
    (d / "q6_basis.json").write_text(
        json.dumps(
            {
                "passed": True,
                "status": "FROZEN",
                "npz": "q6_basis.npz",
                "npz_sha256": sha,
                "identity": ident,
            }
        )
    )
    if corrupt:
        (d / "q6_basis.npz").write_bytes(b"basis-bytes-corrupted")
    return sha


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "{",
        "{}",
        json.dumps({"passed": False, "identity": {}}),
        json.dumps({"passed": True}),
        json.dumps({"passed": True, "identity": {"config_sha256": "wrong"}}),
    ],
    ids=[
        "absent",
        "empty",
        "malformed",
        "empty-object",
        "failed",
        "no-identity",
        "mismatched",
    ],
)
def test_invalid_dacc_smoke_markers_never_satisfy_the_gate(tmp_path, content):
    if content is not None:
        (tmp_path / "dacc-gate-smoke-PASS.json").write_text(content)
    assert not _done(tmp_path, "dacc-gate-smoke")
    # and its dependents are not launched
    cands, notes = M.launch_candidates(tmp_path, JOBS, skip=set(), live={}, state={})
    assert "dacc-t2048" not in [j.name for j, _ in cands]


def test_a_stale_pass_from_other_generation_code_is_rejected(tmp_path):
    stale = {**_dacc_identity(), "implementation_id": "0" * 64}
    (tmp_path / "dacc-gate-smoke-PASS.json").write_text(
        json.dumps({"passed": True, "identity": stale})
    )
    assert not _done(tmp_path, "dacc-gate-smoke")


def test_a_true_pass_with_the_expected_identity_unblocks_dependents(tmp_path):
    (tmp_path / "dacc-gate-smoke-PASS.json").write_text(
        json.dumps({"passed": True, "identity": _dacc_identity()})
    )
    assert _done(tmp_path, "dacc-gate-smoke")
    cands, _ = M.launch_candidates(tmp_path, JOBS, skip=set(), live={}, state={})
    assert "dacc-t2048" in [j.name for j, _ in cands]


def test_a_corrupted_basis_behind_a_valid_header_is_not_a_finished_fit(tmp_path):
    _write_basis(tmp_path, corrupt=True)
    assert not _done(tmp_path, "q6-basis-fit")
    cands, _ = M.launch_candidates(tmp_path, JOBS, skip=set(), live={}, state={})
    names = [j.name for j, _ in cands]
    assert "q6-basis-fit" in names and "capture-smoke" not in names


def test_a_basis_of_another_identity_is_rejected_and_a_true_one_accepted(tmp_path):
    other = tmp_path / "other"
    _write_basis(other, wrong_identity=True)
    assert not _done(other, "q6-basis-fit")
    _write_basis(tmp_path)
    assert _done(tmp_path, "q6-basis-fit")


def test_capture_smoke_identity_must_include_the_verified_basis(tmp_path):
    sha = _write_basis(tmp_path)
    cfg = _yaml(ROOT / BY["capture-smoke"].config)
    good = identity(
        "capture",
        cfg,
        model_revision=_revision(cfg, ROOT),
        reference_run=cfg["reference_run_name"],
        basis_npz_sha256=sha,
    )
    p = tmp_path / "capture-smoke-PASS.json"
    p.write_text(
        json.dumps({"passed": True, "identity": {**good, "basis_npz_sha256": "x"}})
    )
    assert not _done(tmp_path, "capture-smoke")
    p.write_text(json.dumps({"passed": True, "identity": good}))
    assert _done(tmp_path, "capture-smoke")
    # corrupting the basis afterwards invalidates the smoke too
    (tmp_path / "q6-basis-fit/basis/q6_basis.npz").write_bytes(b"x")
    assert not _done(tmp_path, "capture-smoke")


def test_historical_markers_keep_their_scoped_legacy_rule(tmp_path):
    """Only these completed, pre-R.15 artifact markers use the legacy rule; any
    NEW marker job must declare a typed gate (this list is closed on purpose)."""
    legacy = {j.name for j in JOBS if j.progress_kind == "marker" and not j.gate}
    assert legacy == {
        "rkv-smoke",
        "shape-p2v2-restricted",
        "shape-sawtooth-i16",
        "shape-sawtooth-i64",
        "shape-sawtooth-i256",
        "onset-surface-q2",
    }


def _fork_precision_output(runs, **over):
    cfg = _yaml(ROOT / BY["fork-precision"].config)
    ident = identity(
        "fork_precision",
        cfg,
        model_revision=_revision(cfg, ROOT),
        reference_run=cfg["reference_run_name"],
    )
    d = {
        "passed": True,
        "provenance": ident,
        "diagnosis": {t: {"n_compared": 3} for t in cfg["dtypes"]},
        "floor": {f"p{i}": {"256": {"tv": 0.0}} for i in range(10)},
    }
    d.update(over)
    out = runs / "fork-precision" / "fork_precision.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d))


def test_fork_precision_gate_needs_every_registered_part(tmp_path):
    assert not _done(tmp_path, "fork-precision")  # absent
    _fork_precision_output(tmp_path)
    assert _done(tmp_path, "fork-precision")
    for bad in (
        {"passed": False},
        {"provenance": {"kind": "fork_precision"}},
        {"diagnosis": {"bfloat16": {"n_compared": 3}}},  # fp32 missing
        {"floor": {"p0": {"256": {}}}},  # 1 of 10 traces
    ):
        runs = tmp_path / str(len(list(tmp_path.iterdir())))
        _fork_precision_output(runs, **bad)
        assert not _done(runs, "fork-precision"), bad

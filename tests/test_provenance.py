"""Experiment identity and gate markers (R.13): the persistence layer, no GPU."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from kvcache.provenance import (
    GENERATION_SOURCES,
    config_sha256,
    gate_accepted,
    implementation_id,
    write_gate,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]


def test_a_failure_after_a_pass_retires_the_pass(tmp_path):
    write_gate(tmp_path, "g-PASS.json", "g-FAIL.json", {"passed": True})
    assert gate_accepted(tmp_path / "g-PASS.json")
    write_gate(tmp_path, "g-PASS.json", "g-FAIL.json", {"passed": False})
    assert not (tmp_path / "g-PASS.json").exists()
    assert (tmp_path / "g-FAIL.json").exists()
    assert list(tmp_path.glob("g-PASS.json.superseded-*"))  # kept, not deleted
    assert not gate_accepted(tmp_path / "g-PASS.json")


def test_gate_acceptance_requires_passed_and_matching_identity(tmp_path):
    p = write_json_atomic(tmp_path / "m.json", {"passed": True, "identity": {"a": 1}})
    assert gate_accepted(p, {"a": 1})
    assert not gate_accepted(p, {"a": 2})
    (tmp_path / "torn.json").write_text('{"passed": tr')
    assert not gate_accepted(tmp_path / "torn.json")
    assert not list(tmp_path.glob(".*tmp*"))  # atomic writes leave no temp


def test_implementation_id_tracks_generation_code_not_readers(tmp_path):
    for rel in {r for v in GENERATION_SOURCES.values() for r in v}:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    before = implementation_id("capture", tmp_path)
    (tmp_path / "scripts" / "report_capture.py").write_text("# reader edit\n")
    assert implementation_id("capture", tmp_path) == before
    cap = tmp_path / "src/kvcache/capture.py"
    cap.write_text(cap.read_text() + "\n# changed\n")
    assert implementation_id("capture", tmp_path) != before
    assert config_sha256({"a": 1, "b": 2}) == config_sha256({"b": 2, "a": 1})


def test_the_scheduler_rejects_failed_or_torn_markers(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import nightly_autorun as M

    (tmp_path / "ok.json").write_text(json.dumps({"passed": True}))
    (tmp_path / "legacy.json").write_text(json.dumps({"verdict": "x"}))
    (tmp_path / "failed.json").write_text(json.dumps({"passed": False}))
    (tmp_path / "torn.json").write_text("{")
    assert M._marker_ok(tmp_path / "ok.json")
    assert M._marker_ok(tmp_path / "legacy.json")  # historical artifact markers
    assert not M._marker_ok(tmp_path / "failed.json")
    assert not M._marker_ok(tmp_path / "torn.json")
    assert not M._marker_ok(tmp_path / "absent.json")

"""The committed WP4 smoke bundle validates on CPU, and tampering is caught
(Meeting 5 review, G0.1)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/diag/paper_audit/wp4_smoke"


def _run(bundle: Path) -> tuple[int, dict]:
    p = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_smoke_bundle.py"),
            "--bundle",
            str(bundle),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return p.returncode, json.loads(p.stdout)


def test_committed_bundle_is_valid():
    code, v = _run(BUNDLE)
    assert code == 0 and v["verdict"] == "VALID", v
    assert v["records"] == {"replay": 9, "identity": 3, "generation": 3}


def _copy(tmp_path) -> Path:
    dst = tmp_path / "b"
    shutil.copytree(BUNDLE, dst)
    return dst


def _rehash(b: Path) -> None:
    import hashlib

    m = json.loads((b / "bundle_manifest.json").read_text())
    for name in m["files_sha256"]:
        m["files_sha256"][name] = hashlib.sha256((b / name).read_bytes()).hexdigest()
    (b / "bundle_manifest.json").write_text(json.dumps(m))


def test_a_changed_byte_is_a_hash_mismatch(tmp_path):
    b = _copy(tmp_path)
    (b / "protsel-smoke.run_log.txt").write_text("edited")
    code, v = _run(b)
    assert code == 2 and any("hash mismatch" in p for p in v["problems"])


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda r: r.update(failures=["x"]), "lists failures"),
        (lambda r: r.update(status="FAIL"), "passed/PASS"),
        (lambda r: r["records"][0].update(pre_eviction_max_tv=0.5), "re-applied"),
        (
            lambda r: r["identity"].update(implementation_id="0" * 64),
            "identity differs",
        ),
        (lambda r: r["records"].pop(), "records, manifest says"),
    ],
    ids=["failures", "status", "check", "identity", "count"],
)
def test_record_tampering_is_invalid_even_with_fresh_hashes(tmp_path, mutate, needle):
    b = _copy(tmp_path)
    p = b / "protected-selectors-smoke-PASS.json"
    rec = json.loads(p.read_text())
    mutate(rec)
    p.write_text(json.dumps(rec))
    _rehash(b)
    code, v = _run(b)
    assert code == 2 and any(needle in x for x in v["problems"]), v

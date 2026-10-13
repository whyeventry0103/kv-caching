"""BBH logical_deduction_five_objects loader: refuses until a person pins the
revision; verifies the git blob hash; validates every target."""

from __future__ import annotations

import json

import pytest

from kvcache import bbh


def _file(targets):
    return json.dumps(
        {
            "examples": [
                {"input": f"q{i} Options: (A) x", "target": t}
                for i, t in enumerate(targets)
            ]
        }
    ).encode()


def test_git_blob_sha1_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert bbh.git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_parse_checks_blob_and_targets():
    data = _file(["(A)", "(E)"])
    probs = bbh.parse_bbh(data, expected_blob=bbh.git_blob_sha1(data))
    assert [p.pid for p in probs] == ["bbh_lde5/0", "bbh_lde5/1"]
    assert probs[1].gold_answer == "(E)"
    with pytest.raises(ValueError, match="not the pinned"):
        bbh.parse_bbh(data, expected_blob="0" * 40)
    bad = _file(["(F)"])
    with pytest.raises(ValueError, match="target"):
        bbh.parse_bbh(bad, expected_blob=bbh.git_blob_sha1(bad))


def test_load_refuses_while_unpinned(tmp_path):
    assert bbh.PINNED_COMMIT == "TODO(human)"  # a person pins it, never a guess
    path = tmp_path / "f.json"
    path.write_bytes(_file(["(A)"]))
    with pytest.raises(RuntimeError, match="not pinned"):
        bbh.load_bbh(path)


def test_load_after_pin_checks_count(tmp_path, monkeypatch):
    data = _file(["(A)"] * 3)
    path = tmp_path / "f.json"
    path.write_bytes(data)
    monkeypatch.setattr(bbh, "PINNED_COMMIT", "a" * 40)
    monkeypatch.setattr(bbh, "PINNED_BLOB", bbh.git_blob_sha1(data))
    with pytest.raises(ValueError, match="expected 250"):
        bbh.load_bbh(path)

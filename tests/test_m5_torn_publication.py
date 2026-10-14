"""A stop DURING a live append must recover from the intact pair journal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_m5_publish as tp  # noqa: E402
import test_run_grid_m5 as tm  # noqa: E402

from kvcache import m5_publish as mp  # noqa: E402


@pytest.mark.parametrize("target", ["row", "sidecar"])
def test_torn_live_append_recovers_journal_without_regeneration(
    tmp_path, monkeypatch, target
):
    cfg, out, _ = tm._setup(tmp_path, monkeypatch)
    generated = tp._counted(monkeypatch)
    original = mp._publish_one
    injected = {}

    def torn_append(path, record):
        if ("token_ids" in path.parts) == (target == "sidecar"):
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(record, sort_keys=True)
            injected["line"] = text[: len(text) // 2]
            with path.open("a", encoding="utf-8") as stream:
                stream.write(injected["line"])
            raise OSError("injected interruption during live JSONL append")
        return original(path, record)

    monkeypatch.setattr(mp, "_publish_one", torn_append)
    with pytest.raises(OSError, match="during live JSONL append"):
        tm._run(monkeypatch, cfg)
    assert generated["generations"] == 1
    pending = [
        path
        for path in (out / mp.JOURNAL).rglob("*.json")
        if not path.name.endswith(".published.json")
    ]
    assert len(pending) == 1
    saved_row = json.loads(pending[0].read_text())["row"]

    monkeypatch.setattr(mp, "_publish_one", original)
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    rows = tp._pairs_ok(out, 100)
    assert generated["generations"] == 100
    assert next(row for row in rows if row["pid"] == saved_row["pid"]) == saved_row
    quarantined = [
        json.loads(line)
        for path in (out / mp.QUARANTINE).rglob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["line"] == injected["line"]
    assert quarantined[0]["reason"] == "torn or unreadable line"

    # A second resume neither regenerates nor duplicates quarantine evidence.
    assert tm._run(monkeypatch, cfg, "--resume") == 0
    assert generated["generations"] == 100
    tp._pairs_ok(out, 100)
    assert (
        sum(
            len(path.read_text().splitlines())
            for path in (out / mp.QUARANTINE).rglob("*.jsonl")
        )
        == 1
    )

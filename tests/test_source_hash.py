from kvcache.source_hash import source_tree_hash


def test_source_hash_tracks_source_but_ignores_remote_artifacts(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = source_tree_hash(tmp_path)

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "output.jsonl").write_text("remote", encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.local.json").write_text(
        '{"permissions": {"allow": []}}', encoding="utf-8"
    )
    assert source_tree_hash(tmp_path) == first

    source.write_text("value = 2\n", encoding="utf-8")
    assert source_tree_hash(tmp_path) != first

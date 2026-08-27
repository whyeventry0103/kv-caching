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
    (tmp_path / ".venv-next").mkdir()
    (tmp_path / ".venv-next" / "python").write_text("binary", encoding="utf-8")
    assert source_tree_hash(tmp_path) == first

    source.write_text("value = 2\n", encoding="utf-8")
    assert source_tree_hash(tmp_path) != first


def test_source_tree_hash_covers_docs_results_but_not_the_runs_tree(tmp_path):
    """Regression: the exclusions must be anchored at the repo root.

    `results` was matched at ANY depth, so `docs/results/` -- PAPER_FINDINGS.md
    (the results record, #3 in the authority order), RESULTS_PLAYBOOK.md,
    RESULTS.md -- was silently excluded from the source-tree hash. The parity
    gate is what certifies local and remote hold the same source, so it was
    blind to divergence in exactly the document whose job is to be the shared
    record: PAPER_FINDINGS could differ by 48 lines while parity reported the
    source trees matched.

    `sync_source.sh` already anchors these (`--exclude='/results/'`), so the
    hash must cover exactly what sync transfers.
    """
    from kvcache.source_hash import source_tree_hash

    (tmp_path / "docs" / "results").mkdir(parents=True)
    (tmp_path / "results" / "diag").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    doc = tmp_path / "docs" / "results" / "PAPER_FINDINGS.md"
    doc.write_text("original", encoding="utf-8")
    (tmp_path / "results" / "diag" / "big.json").write_text("{}", encoding="utf-8")
    (tmp_path / "src" / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")

    before = source_tree_hash(tmp_path)

    # A change under docs/results/ MUST move the hash.
    doc.write_text("changed", encoding="utf-8")
    assert source_tree_hash(tmp_path) != before, (
        "docs/results/ is invisible to the source-tree hash; the parity gate "
        "cannot see PAPER_FINDINGS diverge"
    )

    # A change under the top-level results/ tree must NOT move it.
    doc.write_text("original", encoding="utf-8")
    assert source_tree_hash(tmp_path) == before
    (tmp_path / "results" / "diag" / "big.json").write_text(
        '{"a": 1}', encoding="utf-8"
    )
    assert (
        source_tree_hash(tmp_path) == before
    ), "the generated results/ tree must stay excluded"

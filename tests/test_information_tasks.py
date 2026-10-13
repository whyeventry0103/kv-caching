"""Controlled target/nuisance task generator (Meeting 5, M5-C), with a toy
word-level tokenizer: complete 16-variant grids, identical positions across
variants, counterbalanced ages, leak screening, and block-level splits."""

from __future__ import annotations

import re

import pytest

from kvcache import information_tasks as it

VOCAB: dict[str, int] = {}


def encode(text: str) -> list[int]:
    return [VOCAB.setdefault(w, len(VOCAB)) for w in re.findall(r"\w+|[^\w\s]", text)]


VALUES = ["alpha", "beta", "gamma", "delta"]
ALIASES = ["zorp", "quix", "blen", "trag"]
SPEC = it.TaskSpec(body_tokens=120, early_offset=10, late_offset=80)


def _passages(n=6, length=60, extra=None):
    out = [
        it.Passage(
            f"trace{i}", tuple(encode(" ".join(f"f{i}x{j}" for j in range(length))))
        )
        for i in range(n)
    ]
    return out + (extra or [])


def _blocks(n=4, passages=None):
    return it.make_blocks(
        n,
        design_seed=7,
        aliases=ALIASES,
        passages=passages or _passages(),
        encode=encode,
        values=VALUES,
        spec=SPEC,
    )


def test_encodings_check_passes_and_catches_each_problem():
    assert it.check_encodings(encode, VALUES, ALIASES, SPEC) == []
    bad = it.check_encodings(encode, ["two words", *VALUES[1:]], ALIASES, SPEC)
    assert any("2 tokens" in b for b in bad)
    assert any("lengths differ" in b for b in bad)
    assert it.check_encodings(encode, VALUES[:3], ALIASES, SPEC) == [
        "need 4 distinct values"
    ]
    leaky = it.TaskSpec(query=" Question: is {alias} alpha? Answer:")
    assert any(
        "contains a value token" in b
        for b in it.check_encodings(encode, VALUES, ALIASES, leaky)
    )


def test_blocks_are_deterministic_and_counterbalanced():
    a, b = _blocks(), _blocks()
    assert a == b
    assert [x.s_first for x in a] == [True, False, True, False]
    assert all(x.s_alias != x.n_alias for x in a)
    assert len({x.block_id for x in a}) == 4


def test_sixteen_variants_differ_only_in_the_two_value_tokens():
    passages = _passages()
    block = _blocks(passages=passages)[1]  # N first
    vs = it.all_variants(
        block, passages=passages, encode=encode, values=VALUES, spec=SPEC
    )
    assert sorted((v.s, v.n) for v in vs) == [
        (s, n) for s in range(4) for n in range(4)
    ]
    assert {len(v.body_ids) for v in vs} == {120}
    assert len({v.prompt_ids for v in vs}) == 1
    assert len({tuple(v.query_ids.items()) for v in vs}) == 1
    assert vs[0].fact_spans == {"n": (10, 14), "s": (80, 84)}
    value_ids = {encode(" " + x)[0] for x in VALUES}
    base = vs[0].body_ids
    for v in vs:
        diff = [
            i for i, (x, y) in enumerate(zip(base, v.body_ids, strict=True)) if x != y
        ]
        assert set(diff) <= {12, 82}  # the value token of each fact
        assert v.body_ids[82] == encode(" " + VALUES[v.s])[0]
        assert v.body_ids[12] == encode(" " + VALUES[v.n])[0]
        outside = [t for i, t in enumerate(v.body_ids) if i not in (12, 82)]
        assert not set(outside) & value_ids  # values appear only in the facts
        for q in v.query_ids.values():
            assert not set(q) & value_ids


def test_leaky_filler_is_skipped_and_logged():
    leaky = it.Passage("leaky", tuple(encode("we found beta here " * 20)))
    alias_leak = it.Passage("aliasy", tuple(encode("zorp quix blen trag " * 20)))
    passages = [leaky, alias_leak, *_passages()]
    blocks = _blocks(n=6, passages=passages)
    used = {src for b in blocks for src, _, _ in b.filler}
    assert "leaky" not in used  # contains a value token
    assert "aliasy" not in used  # contains every block's aliases
    # a block stops once filled, so it logs only the leaky passages it reached
    assert any("leaky" in b.skipped_sources for b in blocks)
    assert any("aliasy" in b.skipped_sources for b in blocks)


def test_insufficient_screened_filler_refuses():
    with pytest.raises(ValueError, match="filler too short"):
        _blocks(passages=[it.Passage("tiny", tuple(encode("a b c")))])


def test_spec_refusals():
    with pytest.raises(ValueError, match="does not fit"):
        it.make_blocks(
            1,
            design_seed=0,
            aliases=ALIASES,
            passages=_passages(),
            encode=encode,
            values=VALUES,
            spec=it.TaskSpec(body_tokens=100, early_offset=10, late_offset=99),
        )
    with pytest.raises(ValueError, match="two aliases"):
        it.make_blocks(
            1,
            design_seed=0,
            aliases=ALIASES[:1],
            passages=_passages(),
            encode=encode,
            values=VALUES,
            spec=SPEC,
        )


def test_split_is_disjoint_complete_and_stable():
    blocks = _blocks(n=10)
    sp = it.split_blocks(blocks, n_calibration=4, design_seed=7)
    assert len(sp["calibration"]) == 4 and len(sp["confirmation"]) == 6
    assert not set(sp["calibration"]) & set(sp["confirmation"])
    assert set(sp["calibration"]) | set(sp["confirmation"]) == {
        b.block_id for b in blocks
    }
    assert sp == it.split_blocks(blocks, n_calibration=4, design_seed=7)
    with pytest.raises(ValueError):
        it.split_blocks(blocks, n_calibration=0, design_seed=7)

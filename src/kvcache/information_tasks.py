"""Controlled target/nuisance memory task for the information probe (Meeting 5,
M5-C; prepared 2026-10-14, before any probe outcome exists).

Scientific rationale (review section 10.1). To measure task information without
the deterministic-question trap, the fact must be randomized independently of
everything the reader sees except the cache. Each context block u carries two
independent four-valued variables: a target S and a nuisance N. All 16 (S, N)
combinations are generated for every block, with equal probability, so the
reader (``kvcache.task_information``) can marginalize N exactly.

Construction, in TOKEN space so positions are identical across the 16 variants:

    prompt (protected; task + allowed symbols, NO values)
    body   = filler | fact A | filler | fact B | filler     (decode path, evictable)
    query  = "what is <alias>?" for S, or for N           (readout, retained)

* Facts are "<alias> is <value>." with nonce aliases; every value encoding must
  give the SAME token length, and every value must be a single token, for the
  pinned tokenizer (``check_encodings``), so a variant changes value tokens only.
* Which variable comes first alternates by block (counterbalanced age): in even
  blocks S is the early fact, in odd blocks N is.
* Filler is taken from supplied passages (e.g. real full-cache reasoning text,
  with the source trace hash retained), chosen by a SHA-256 draw keyed on the
  block, never on any outcome. A passage that contains a value token or an alias
  token sequence is skipped (leak screen) and the skip is logged.
* Splits are by whole block: no (S, N) variant of a calibration block can appear
  in confirmation (``split_blocks``).

Nothing here touches a model; a runner feeds ``Variant`` token ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

K = 4
Encode = Callable[[str], list[int]]


@dataclass(frozen=True)
class TaskSpec:
    """Frozen design of one regime (the primary is 1,024 body tokens at B=256)."""

    body_tokens: int = 1024
    early_offset: int = 64  # body position of the first fact
    late_offset: int = 768  # body position of the second fact
    instruction: str = (
        "You will read notes containing codes. Afterwards you will be asked for "
        "one code. Answer with exactly one of: {symbols}."
    )
    fact: str = " {alias} is {value}."
    query: str = " Question: what is {alias}? Answer:"


@dataclass(frozen=True)
class Passage:
    source: str  # e.g. trace hash of the full-cache reasoning it came from
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Block:
    block_id: str
    s_alias: str
    n_alias: str
    s_first: bool
    filler: tuple[tuple[str, int, int], ...]  # (source, start, length) segments
    skipped_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Variant:
    block_id: str
    s: int
    n: int
    prompt_ids: tuple[int, ...]
    body_ids: tuple[int, ...]
    query_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fact_spans: dict[str, tuple[int, int]] = field(default_factory=dict)


def draw(key: str, n: int) -> int:
    """Stateless deterministic draw in [0, n) keyed by a string."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % n


def check_encodings(
    encode: Encode, values: Sequence[str], aliases: Sequence[str], spec: TaskSpec
) -> list[str]:
    """Every reason the symbol/alias set is unusable for this tokenizer.

    Each value must be ONE token in answer position; every fact must have the
    same token length for every value with a given alias, and all aliases must
    give the same fact length (so ages are identical across blocks); the
    instruction and queries must contain no value token.
    """
    out: list[str] = []
    if len(values) != K or len(set(values)) != K:
        return [f"need {K} distinct values"]
    value_ids = []
    for v in values:
        ids = encode(" " + v)
        if len(ids) != 1:
            out.append(f"value {v!r} is {len(ids)} tokens in answer position")
        value_ids.extend(ids)
    lengths = {
        (a, v): len(encode(spec.fact.format(alias=a, value=v)))
        for a in aliases
        for v in values
    }
    if len(set(lengths.values())) != 1:
        out.append(f"fact token lengths differ: {sorted(set(lengths.values()))}")
    for a in aliases:
        q = encode(spec.query.format(alias=a))
        if set(q) & set(value_ids):
            out.append(f"query for {a!r} contains a value token")
    # the instruction lists the allowed symbols (all four, equally); it must
    # never name an alias, or it would pair a symbol with a variable
    for a in aliases:
        if a in spec.instruction:
            out.append(f"instruction mentions alias {a!r}")
    return out


def _contains(hay: Sequence[int], needle: Sequence[int]) -> bool:
    n = len(needle)
    return any(tuple(hay[i : i + n]) == tuple(needle) for i in range(len(hay) - n + 1))


def leaks(
    ids: Sequence[int], value_ids: Sequence[int], alias_ids: Sequence[Sequence[int]]
) -> bool:
    return bool(set(ids) & set(value_ids)) or any(_contains(ids, a) for a in alias_ids)


def make_blocks(
    n_blocks: int,
    *,
    design_seed: int,
    aliases: Sequence[str],
    passages: Sequence[Passage],
    encode: Encode,
    values: Sequence[str],
    spec: TaskSpec | None = None,
) -> list[Block]:
    """Deterministic blocks: alias pair, counterbalanced order, screened filler."""
    spec = spec or TaskSpec()
    if len(aliases) < 2:
        raise ValueError("need at least two aliases")
    value_ids = [encode(" " + v)[0] for v in values]
    fact_len = len(encode(spec.fact.format(alias=aliases[0], value=values[0])))
    need = spec.body_tokens - 2 * fact_len
    if need <= 0 or not 0 <= spec.early_offset < spec.late_offset:
        raise ValueError("body too short for two facts at the declared offsets")
    if spec.late_offset + fact_len > spec.body_tokens:
        raise ValueError("late fact does not fit in the body")
    blocks = []
    for b in range(n_blocks):
        key = f"{design_seed}|block|{b}"
        i = draw(key + "|s", len(aliases))
        j = draw(key + "|n", len(aliases) - 1)
        s_alias, n_alias = aliases[i], [a for k, a in enumerate(aliases) if k != i][j]
        alias_ids = [encode(" " + s_alias), encode(" " + n_alias)]
        filler: list[tuple[str, int, int]] = []
        skipped: list[str] = []
        have = 0
        start = draw(key + "|passage", len(passages))
        for step in range(len(passages)):
            p = passages[(start + step) % len(passages)]
            if leaks(p.token_ids, value_ids, alias_ids):
                skipped.append(p.source)
                continue
            take = min(len(p.token_ids), need - have)
            filler.append((p.source, 0, take))
            have += take
            if have == need:
                break
        if have < need:
            raise ValueError(f"block {b}: screened filler too short ({have}/{need})")
        blocks.append(
            Block(
                block_id=f"b{design_seed}-{b:04d}",
                s_alias=s_alias,
                n_alias=n_alias,
                s_first=(b % 2 == 0),
                filler=tuple(filler),
                skipped_sources=tuple(skipped),
            )
        )
    return blocks


def render(
    block: Block,
    s: int,
    n: int,
    *,
    passages: Sequence[Passage],
    encode: Encode,
    values: Sequence[str],
    spec: TaskSpec | None = None,
) -> Variant:
    """Token ids of one (S, N) variant; only the two value tokens vary."""
    spec = spec or TaskSpec()
    by_source = {p.source: p for p in passages}
    filler = [
        t
        for src, a, length in block.filler
        for t in by_source[src].token_ids[a : a + length]
    ]
    fact_s = encode(spec.fact.format(alias=block.s_alias, value=values[s]))
    fact_n = encode(spec.fact.format(alias=block.n_alias, value=values[n]))
    first, second = (fact_s, fact_n) if block.s_first else (fact_n, fact_s)
    e, l_ = spec.early_offset, spec.late_offset
    gap = l_ - e - len(first)
    body = filler[:e] + first + filler[e : e + gap] + second + filler[e + gap :]
    if len(body) != spec.body_tokens:
        raise RuntimeError(f"body is {len(body)} tokens, not {spec.body_tokens}")
    spans = {
        ("s" if block.s_first else "n"): (e, e + len(first)),
        ("n" if block.s_first else "s"): (l_, l_ + len(second)),
    }
    return Variant(
        block_id=block.block_id,
        s=s,
        n=n,
        prompt_ids=tuple(encode(spec.instruction.format(symbols=", ".join(values)))),
        body_ids=tuple(body),
        query_ids={
            "s": tuple(encode(spec.query.format(alias=block.s_alias))),
            "n": tuple(encode(spec.query.format(alias=block.n_alias))),
        },
        fact_spans=spans,
    )


def all_variants(block: Block, **kw: Any) -> list[Variant]:
    return [render(block, s, n, **kw) for s in range(K) for n in range(K)]


def ids_sha256(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()


def input_manifest(
    blocks: Sequence[Block],
    variants: Mapping[str, Sequence[Variant]],
    *,
    passages: Sequence[Passage],
    symbol_ids: Sequence[int],
) -> dict[str, Any]:
    """The ACTUAL selected inputs of a run (review of 5616630, R2): every
    filler passage (source, length, token-id hash), every block definition
    (aliases, order, filler segments, skipped sources) and the token-id hashes
    of every rendered variant's instruction, body and queries. Hashing this --
    not the config, the block ids or the code -- is what binds a run to its
    inputs: changed filler with the same config changes the manifest."""
    return {
        "schema": "info-input-manifest-v1",
        "symbol_ids": [int(t) for t in symbol_ids],
        "passages": [
            {
                "source": p.source,
                "n_tokens": len(p.token_ids),
                "token_ids_sha256": ids_sha256(p.token_ids),
            }
            for p in passages
        ],
        "blocks": [
            {
                "block_id": b.block_id,
                "s_alias": b.s_alias,
                "n_alias": b.n_alias,
                "s_first": b.s_first,
                "filler": [list(seg) for seg in b.filler],
                "skipped_sources": list(b.skipped_sources),
            }
            for b in blocks
        ],
        "variants": {
            b.block_id: [
                {
                    "s": v.s,
                    "n": v.n,
                    "prompt_sha256": ids_sha256(v.prompt_ids),
                    "body_sha256": ids_sha256(v.body_ids),
                    "query_sha256": {
                        q: ids_sha256(ids) for q, ids in sorted(v.query_ids.items())
                    },
                    "fact_spans": {k: list(s) for k, s in sorted(v.fact_spans.items())},
                }
                for v in variants[b.block_id]
            ]
            for b in blocks
        },
    }


def split_blocks(
    blocks: Sequence[Block], *, n_calibration: int, design_seed: int
) -> dict[str, list[str]]:
    """Disjoint calibration/confirmation block ids by a hash order (no outcome)."""
    ids = sorted(
        (b.block_id for b in blocks),
        key=lambda x: hashlib.sha256(f"{design_seed}|split|{x}".encode()).hexdigest(),
    )
    if not 0 < n_calibration < len(ids):
        raise ValueError("calibration must be a proper, non-empty subset")
    return {
        "calibration": sorted(ids[:n_calibration]),
        "confirmation": sorted(ids[n_calibration:]),
    }

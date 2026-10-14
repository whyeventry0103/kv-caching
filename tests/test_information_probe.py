"""Information-probe core on CPU (review of 6b409001, R3): a stub model that
genuinely retrieves "<alias> is <value>" facts from whatever its cache still
holds, a recency-eviction stub press, real generator variants and the real
reader. Full cache gives 2 bits; recency eviction loses the EARLY fact only;
shared clones and eviction during readout trip their controls and make the
production read INVALID."""

from __future__ import annotations

import copy
import math
import re

import pytest

from kvcache import information_probe as ip
from kvcache import information_tasks as it
from kvcache import task_information as ti

VOCAB: dict[str, int] = {}


def encode(text: str) -> list[int]:
    return [VOCAB.setdefault(w, len(VOCAB)) for w in re.findall(r"\w+|[^\w\s]", text)]


VALUES = ["alpha", "beta", "gamma", "delta"]
ALIASES = ["zorp", "quix", "blen", "trag"]
SPEC = it.TaskSpec(body_tokens=120, early_offset=10, late_offset=90)
PASSAGES = [
    it.Passage(f"tr{i}", tuple(encode(" ".join(f"w{i}x{j}" for j in range(80)))))
    for i in range(4)
]
SYMBOLS = [encode(" " + v)[0] for v in VALUES]
IS_ID = encode(" is")[0]
for w in ("Question", ":", "what", "?", "Answer"):
    encode(w)
VSIZE = 400
for i in range(VSIZE):
    encode(f"pad{i}")


class Stub:
    """Cache = list of (position, token). Logits: after a query for alias X,
    +30 on value v iff the cache holds the triple (X, is, v); else flat."""

    def __init__(self, share_clones=False):
        self.share = share_clones

    def new_cache(self):
        return {"items": [], "next": 0}

    def clone(self, cache):
        return cache if self.share else copy.deepcopy(cache)

    def forward(self, ids, cache):
        for t in ids:
            cache["items"].append((cache["next"], t))
            cache["next"] += 1
        toks = [t for _, t in cache["items"]]
        logits = [0.0] * len(VOCAB)
        alias = toks[-4] if len(toks) >= 4 else None  # "what is X ? Answer :"
        for i in range(len(toks) - 2):
            if toks[i] == alias and toks[i + 1] == IS_ID and toks[i + 2] in SYMBOLS:
                logits[toks[i + 2]] += 30.0
        logits[SYMBOLS[0]] += 1e-3 * len(toks)  # makes shared state visible
        return logits

    def log_softmax(self, logits):
        m = max(logits)
        z = m + math.log(sum(math.exp(x - m) for x in logits))
        return [x - z for x in logits]


class Recency:
    """Keep the prompt plus the most recent (budget - prompt) entries."""

    def __init__(self, cache_ref, prompt_len, budget, honor_disable=True):
        self.cache_ref, self.p, self.b = cache_ref, prompt_len, budget
        self.on, self.prunes, self.honor = False, 0, honor_disable

    def attach(self):
        from contextlib import nullcontext

        return nullcontext()

    def enabled(self, flag):
        outer = self

        class _Ctx:
            def __enter__(self):
                self.prev = outer.on
                outer.on = flag or not outer.honor
                return self

            def __exit__(self, *a):
                outer.on = self.prev

        return _Ctx()

    def after_forward(self, cache):
        items = cache["items"]
        if self.on and len(items) > self.b:
            cache["items"] = items[: self.p] + items[-(self.b - self.p) :]
            self.prunes += 1

    def n_prunes(self):
        return self.prunes

    def retained(self):
        return (
            {0: [[p for p, _ in self.cache_ref["items"]]]} if self.cache_ref else None
        )


class PressedStub(Stub):
    """Applies the active press after each forward, like a decoding press hook."""

    press = None

    def forward(self, ids, cache):
        out = super().forward(ids, cache)
        if self.press is not None:
            self.press.cache_ref = cache
            self.press.after_forward(cache)
        return out


def _variants(blocks):
    return [
        it.all_variants(b, passages=PASSAGES, encode=encode, values=VALUES, spec=SPEC)
        for b in blocks
    ]


def _policies(backend, honor=True):
    def full(seed):
        backend.press = None
        return ip.NoPress()

    def recency(seed):
        p = Recency(None, 0, 40, honor_disable=honor)
        backend.press = p
        return p

    return {"full": full, "recency": recency}


PROV = {
    "implementation_id": "i" * 64,
    "model_revision": "stub",
    "tokenizer_revision": "stub",
    "answer_alphabet_sha256": ip.sha256_json(VALUES),
    "task_layout_sha256": "t" * 64,
    "design_sha256": "d" * 64,
}


def _rows(backend, honor=True, n_blocks=2):
    blocks = it.make_blocks(
        n_blocks,
        design_seed=3,
        aliases=ALIASES,
        passages=PASSAGES,
        encode=encode,
        values=VALUES,
        spec=SPEC,
    )
    rows = []
    for vs in _variants(blocks):
        rows += ip.run_context(
            backend,
            vs,
            policies=_policies(backend, honor),
            eviction_seeds=[0],
            symbol_ids=SYMBOLS,
            provenance=PROV,
        )
    return blocks, rows


def _design(blocks):
    return {
        "mode": "production",
        "expected_provenance": PROV,
        "policies": ["full", "recency"],
        "reference_policy": "full",
        "contexts": [b.block_id for b in blocks],
        "eviction_seeds": [0],
        "queries": ["s", "n"],
    }


def test_full_cache_carries_both_facts_recency_loses_the_early_one():
    backend = PressedStub()
    blocks, rows = _rows(backend)
    assert len(rows) == 2 * 2 * 16 * 2  # blocks x policies x variants x queries
    assert all(all(r["controls"].values()) for r in rows)
    read = ti.read_information_probe(rows, _design(blocks))
    assert read["status"] == "FINAL", read["problems"]
    full, rec = read["policies"]["full"], read["policies"]["recency"]
    assert full["query_s"]["mi_bits"]["mean"] == pytest.approx(2.0, abs=1e-3)
    # block 0: S early (evicted) -> 0 bits; block 1: S late (kept) -> 2 bits
    by_block = {}
    for b in blocks:
        cells = [
            ti.Readout(r["s"], r["n"], tuple(r["symbol_logprobs"]), r["greedy"])
            for r in rows
            if r["policy"] == "recency"
            and r["query"] == "s"
            and r["context_id"] == b.block_id
        ]
        by_block[b.s_first] = ti.mutual_information_bits(cells, "s")
    assert by_block[True] == pytest.approx(0.0, abs=1e-3)
    assert by_block[False] == pytest.approx(2.0, abs=1e-3)
    assert rec["query_s"]["mi_bits"]["mean"] == pytest.approx(1.0, abs=1e-3)
    # retention is logged: the early fact span is gone, the late one kept
    r0 = next(
        r
        for r in rows
        if r["policy"] == "recency" and r["context_id"] == blocks[0].block_id
    )
    assert r0["retained_fraction"]["s"] == 0.0 and r0["retained_fraction"]["n"] == 1.0


def test_shared_clone_state_trips_the_control():
    backend = PressedStub(share_clones=True)
    blocks, rows = _rows(backend, n_blocks=1)
    assert not all(r["controls"]["clone_agreement"] for r in rows)
    assert ti.read_information_probe(rows, _design(blocks))["status"] == "INVALID"


def test_eviction_during_readout_trips_the_control():
    backend = PressedStub()
    blocks, rows = _rows(backend, honor=False, n_blocks=1)
    bad = [r for r in rows if r["policy"] == "recency"]
    assert not all(r["controls"]["no_readout_eviction"] for r in bad)
    assert ti.read_information_probe(rows, _design(blocks))["status"] == "INVALID"


def test_resume_into_other_provenance_is_refused():
    _, rows = _rows(PressedStub(), n_blocks=1)
    assert ip.resume_problems(rows, PROV) == []
    assert ip.resume_problems(rows, {**PROV, "design_sha256": "x"})

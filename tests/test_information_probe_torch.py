"""The REAL information-probe path on CPU (review of 6b409001, R3): the
transformers backend with a tiny randomly initialised Qwen2 model (built from a
config -- no download, no weights) and a real kvpress prompt-protected press.
Cache clones agree exactly, the readout evicts nothing, the body pruned, the
full/full control holds, and the rows pass the production reader. Skipped where
torch/transformers/kvpress are absent (the CPU-only gate); run on the host."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
kvpress = pytest.importorskip("kvpress")

from kvcache import information_probe as ip  # noqa: E402
from kvcache import task_information as ti  # noqa: E402
from kvcache.information_tasks import Variant  # noqa: E402
from kvcache.presses.prompt_protect import build_prompt_protected_press  # noqa: E402

SYMBOLS = [10, 11, 12, 13]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    return transformers.Qwen2ForCausalLM(cfg).eval()


def _variant(s, n):
    body = [20 + (i % 90) for i in range(96)]
    body[8], body[70] = SYMBOLS[s], SYMBOLS[n]  # value tokens inside the facts
    return Variant(
        block_id="b0",
        s=s,
        n=n,
        prompt_ids=(1, 2, 3, 4, 5, 6),
        body_ids=tuple(body),
        query_ids={"s": (7, 8, 9), "n": (7, 8, 14)},
        fact_spans={"s": (6, 10), "n": (68, 72)},
    )


def _policies(model):
    def full(seed):
        return ip.NoPress()

    def random_pp(seed):
        p = build_prompt_protected_press(
            "random_pp",
            budget=40,
            compression_interval=8,
            seed=seed,
            kvpress_module=kvpress,
            instrumented=True,
        )
        return ip.KVPressHandle(p, model)

    return {"full": full, "random_pp": random_pp}


def test_real_backend_clones_and_press_controls(model):
    backend = ip.TransformersBackend(model)
    pol = _policies(model)
    res = ip.run_variant(
        backend, pol["random_pp"](0), _variant(1, 2), symbol_ids=SYMBOLS
    )
    for q in ("s", "n"):
        assert res[q]["clone_agreement"] is True
        assert res[q]["no_readout_eviction"] is True
        assert res[q]["n_prunes"] > 0  # the body was actually pruned
        assert all(0.0 <= f <= 1.0 for f in res[q]["retained_fraction"].values())
        assert len(res[q]["symbol_logprobs"]) == 4


def test_real_rows_pass_the_production_reader(model):
    backend = ip.TransformersBackend(model)
    prov = {f: "x" for f in ti.PROVENANCE_FIELDS}
    variants = [_variant(s, n) for s in range(4) for n in range(4)]
    rows = ip.run_context(
        backend,
        variants,
        policies=_policies(model),
        eviction_seeds=[0],
        symbol_ids=SYMBOLS,
        provenance=prov,
    )
    assert all(all(r["controls"].values()) for r in rows)
    read = ti.read_information_probe(
        rows,
        {
            "mode": "production",
            "expected_provenance": prov,
            "policies": ["full", "random_pp"],
            "reference_policy": "full",
            "contexts": ["b0"],
            "eviction_seeds": [0],
            "queries": ["s", "n"],
        },
    )
    assert read["status"] == "FINAL", read["problems"]
    for q in ("query_s", "query_n"):
        mi = read["policies"]["full"][q]["mi_bits"]["mean"]
        assert 0.0 <= mi <= 2.0  # a random model: any value in range is valid

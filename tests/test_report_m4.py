import importlib.util
import json
from pathlib import Path

from pytest import approx

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_m4", ROOT / "scripts" / "report_m4.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_m4_report_uses_accuracy_scale_residual(tmp_path):
    full = tmp_path / "full"
    comp = tmp_path / "comp"
    replay = tmp_path / "replay"
    readout = tmp_path / "readout"
    full_rows = [
        {"pid": f"p{i}", "decoding": {"seed": i}, "correct": True, "truncated": False}
        for i in range(20)
    ]
    comp_rows = [
        {
            "pid": f"p{i}",
            "decoding": {"seed": i},
            "press": "streamingllm",
            "budget": 256,
            "correct": i < 4,
            "truncated": False,
            "n_tokens": 10,
        }
        for i in range(20)
    ]
    replay_rows = [
        {
            "press": "streamingllm",
            "budget": 256,
            "auc_tv": 0.1,
            "tv_profile_path": f"profiles/{i}.json",
            "frac_above": {"p05": 0.2, "p10": 0.1},
            "onset_step": 65,
        }
        for i in range(20)
    ]
    readout_rows = [
        {
            "reference_key": ["math500", f"p{i}", "model", "fullcache", "full", i],
            "condition": {"press": "streamingllm", "budget": 256},
            "readout_correct": i < 9,
            "generated_token_ids": [1, 2],
        }
        for i in range(20)
    ]
    _write(full / "math500" / "rows.jsonl", full_rows)
    _write(comp / "math500" / "rows.jsonl", comp_rows)
    _write(replay / "math500" / "rows.jsonl", replay_rows)
    for i in range(20):
        profile_path = replay / "profiles" / f"{i}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps({"estimate": [0.4, 0.6]}))
    _write(readout / "readout" / "rows.jsonl", readout_rows)

    summary, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        replay_root=replay,
        readout_root=readout,
    )

    assert summary["full_cache"]["accuracy"] == 1.0
    assert cells[0]["compressed_accuracy"] == 0.2
    assert cells[0]["readout_accuracy"] == 0.45
    assert cells[0]["total_change"] == -0.8
    assert cells[0]["readout_change"] == -0.55
    assert cells[0]["unexplained_residual"] == -0.25
    assert cells[0]["signed_flip_mass"]["lost"] == 0.8


def _minimal_inputs(tmp_path):
    full = tmp_path / "full"
    comp = tmp_path / "comp"
    readout = tmp_path / "readout"
    _write(
        full / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "correct": True,
                "truncated": False,
            }
            for i in range(20)
        ],
    )
    _write(
        comp / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "press": "streamingllm",
                "budget": 256,
                "correct": i < 4,
                "truncated": False,
                "n_tokens": 10,
            }
            for i in range(20)
        ],
    )
    _write(
        readout / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["math500", f"p{i}", "model", "fullcache", "full", i],
                "condition": {"press": "streamingllm", "budget": 256},
                "readout_correct": i < 9,
                "generated_token_ids": [1, 2],
            }
            for i in range(20)
        ],
    )
    return full, comp, readout


def test_m4_report_without_replay_still_decomposes(tmp_path):
    full, comp, readout = _minimal_inputs(tmp_path)
    summary, cells = MODULE.build_report(
        fullcache_root=full, compression_root=comp, readout_root=readout
    )
    assert cells[0]["unexplained_residual"] == -0.25
    assert cells[0]["mean_auc_tv"] is None  # TV columns absent without a replay run
    assert summary["runs"]["replay"] is None


def test_m4_reverse_readout_residual_bracket(tmp_path):
    full, comp, readout = _minimal_inputs(tmp_path)
    reverse = tmp_path / "reverse"
    # reverse readout: the COMPRESSED trace (press=streamingllm) read out under full
    # cache; reference_key[3:5] = (streamingllm, 256) keys it to the cell.
    _write(
        reverse / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["math500", f"p{i}", "model", "streamingllm", 256, i],
                "condition": {"press": "fullcache", "budget": "full"},
                "readout_correct": i < 14,
                "generated_token_ids": [1, 2],
            }
            for i in range(20)
        ],
    )
    _, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        reverse_readout_root=reverse,
    )
    cell = cells[0]
    # reverse readout acc = 14/20 = 0.70; reverse_change = 0.70 - 1.0 = -0.30
    assert cell["reverse_readout_accuracy"] == approx(0.70)
    assert cell["reverse_readout_change"] == approx(-0.30)
    # bracket endpoints: forward = total - fwd_readout = -0.25; reverse = the
    # reverse change itself (acc_rev - acc_fc = -0.30), i.e. total minus the
    # on-policy readout change (total - (free - rev)). NOT total - reverse_change,
    # which would be the readout change (-0.50) mislabeled as a residual.
    assert cell["residual_bracket"] == approx([-0.30, -0.25])
    # problem-clustered bootstrap CIs on the endpoints: intervals must be ordered,
    # bounded in [-1, 1], and contain their point estimates on this balanced input.
    f_lo, f_hi = cell["unexplained_residual_ci95"]
    r_lo, r_hi = cell["reverse_residual_ci95"]
    assert -1.0 <= f_lo <= -0.25 <= f_hi <= 1.0
    assert -1.0 <= r_lo <= -0.30 <= r_hi <= 1.0


def test_m4_report_bench_parameter_reads_aime_subtree(tmp_path):
    full = tmp_path / "full"
    comp = tmp_path / "comp"
    readout = tmp_path / "readout"
    _write(
        full / "aime24" / "r.jsonl",
        [
            {
                "pid": f"aime24-{i}",
                "decoding": {"seed": i},
                "correct": True,
                "truncated": False,
            }
            for i in range(10)
        ],
    )
    _write(
        comp / "aime24" / "r.jsonl",
        [
            {
                "pid": f"aime24-{i}",
                "decoding": {"seed": i},
                "press": "rkv",
                "budget": 2048,
                "correct": i < 5,
                "truncated": False,
                "n_tokens": 10,
            }
            for i in range(10)
        ],
    )
    _write(
        readout / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["aime24", f"aime24-{i}", "m", "fullcache", "full", i],
                "condition": {"press": "rkv", "budget": 2048},
                "readout_correct": i < 8,
                "generated_token_ids": [1],
            }
            for i in range(10)
        ],
    )
    summary, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        bench="aime24",
    )
    assert summary["full_cache"]["rows"] == 10
    assert cells[0]["compressed_accuracy"] == 0.5
    assert cells[0]["readout_accuracy"] == 0.8


def _amendment_k_inputs(tmp_path):
    """A cell that is a strict, EASY subset of the FC bank, with undefined rows.

    This is the shape that produced the real defect (Amendment K): the reference
    bank spans problems the cell never ran, and readout / reverse readout are
    defined only on a terminated -- therefore easier -- subset of the cell.

    FC bank: p0..p4 correct, p5..p9 wrong  -> Acc_FC(all rows)  = 0.5
    Cell:    p0..p4 only                   -> Acc_FC(cell)      = 1.0
    """
    full, comp = tmp_path / "full", tmp_path / "comp"
    readout, reverse = tmp_path / "readout", tmp_path / "reverse"
    _write(
        full / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "correct": i < 5,
                "truncated": False,
            }
            for i in range(10)
        ],
    )
    _write(
        comp / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "press": "streamingllm",
                "budget": 256,
                "correct": i < 1,  # 1 of 5 -> 0.2
                "truncated": False,
                "n_tokens": 10,
            }
            for i in range(5)
        ],
    )
    # Readout defined on p0..p3 (p4 undefined), correct on p0,p1 -> 2/4 = 0.5.
    _write(
        readout / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["math500", f"p{i}", "m", "fullcache", "full", i],
                "condition": {"press": "streamingllm", "budget": 256},
                "readout_correct": (i < 2) if i < 4 else None,
                "generated_token_ids": [1],
            }
            for i in range(5)
        ],
    )
    # Reverse defined on p0..p2 only, all correct -> 1.0 on a subset the reference
    # also scores 1.0 on. Against the global 0.5 bank mean this fabricates a +0.5
    # endpoint; against its own population the endpoint is 0.0.
    _write(
        reverse / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["math500", f"p{i}", "m", "streamingllm", "256", i],
                "condition": {"press": "fullcache", "budget": "full"},
                "readout_correct": True if i < 3 else None,
                "generated_token_ids": [1],
            }
            for i in range(5)
        ],
    )
    return full, comp, readout, reverse


def test_amendment_k_reference_is_the_cell_population_not_the_whole_bank(tmp_path):
    full, comp, readout, reverse = _amendment_k_inputs(tmp_path)
    summary, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        reverse_readout_root=reverse,
    )
    cell = cells[0]
    # Provenance keeps the global mean; the cell's reference is its own population.
    assert summary["full_cache"]["accuracy"] == approx(0.5)
    assert cell["full_accuracy_all_rows"] == approx(0.5)
    assert cell["full_accuracy"] == approx(1.0)
    assert cell["reference_population"]["n_conditions"] == 5
    # Pre-Amendment-K this was 0.2 - 0.5 = -0.3, measured against problems the
    # cell never ran.
    assert cell["total_change"] == approx(-0.8)
    assert summary["reference_convention"] == "amendment-k-cell-population"


def test_amendment_k_complete_case_uses_the_defined_subset_as_its_own_reference(
    tmp_path,
):
    full, comp, readout, reverse = _amendment_k_inputs(tmp_path)
    _, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        reverse_readout_root=reverse,
    )
    cell = cells[0]
    assert cell["reference_population"]["n_readout_defined"] == 4
    assert cell["reference_population"]["fc_accuracy_readout_defined"] == approx(1.0)
    assert cell["readout_accuracy"] == approx(0.5)
    # Pre-Amendment-K: 0.5 - 0.5 = 0.0, a readout that looked lossless because the
    # bank mean happened to match an accuracy on an unrelated population.
    assert cell["readout_change"] == approx(-0.5)
    # residual = comp - readout on the SAME 4 conditions: 0.25 - 0.5.
    assert cell["unexplained_residual"] == approx(-0.25)


def test_amendment_k_reverse_endpoint_no_longer_fabricates_a_positive_edge(tmp_path):
    """The 14B failure mode: reverse defined only where the compressed run
    terminated, compared against a bank mean over every condition."""
    full, comp, readout, reverse = _amendment_k_inputs(tmp_path)
    _, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        reverse_readout_root=reverse,
    )
    cell = cells[0]
    assert cell["reference_population"]["n_reverse_defined"] == 3
    assert cell["reverse_readout_accuracy"] == approx(1.0)
    assert cell["reference_population"]["fc_accuracy_reverse_defined"] == approx(1.0)
    # Pre-Amendment-K: 1.0 - 0.5 = +0.5, and the bracket spanned zero for no
    # reason but the mismatch. On its own population the endpoint is exactly 0.
    assert cell["reverse_readout_change"] == approx(0.0)
    assert cell["residual_bracket"] == approx([-0.25, 0.0])


def test_amendment_k_primary_bounds_fix_the_population_and_impute_both_ways(tmp_path):
    full, comp, readout, reverse = _amendment_k_inputs(tmp_path)
    _, cells = MODULE.build_report(
        fullcache_root=full,
        compression_root=comp,
        readout_root=readout,
        reverse_readout_root=reverse,
    )
    bounds = cells[0]["primary_bounds"]
    # 2 defined-correct of 5 conditions; 1 undefined -> [2/5, 3/5].
    assert bounds["readout_accuracy"] == approx([0.4, 0.6])
    assert bounds["readout_change"] == approx([-0.6, -0.4])
    # residual = comp_acc - readout_acc, so the upper readout bound is the more
    # negative residual endpoint.
    assert bounds["residual"] == approx([-0.4, -0.2])
    # 3 defined-correct of 5; 2 undefined -> [3/5, 5/5].
    assert bounds["reverse_readout_accuracy"] == approx([0.6, 1.0])
    assert bounds["reverse_change"] == approx([-0.4, 0.0])
    assert bounds["residual_bracket"] == approx([-0.4, 0.0])


def test_amendment_k_bounds_collapse_to_the_point_when_nothing_is_undefined(tmp_path):
    """Sanity: with every boundary defined and the cell equal to the bank, the
    two layers agree and the imputation interval has zero width."""
    full, comp = tmp_path / "full", tmp_path / "comp"
    readout = tmp_path / "readout"
    _write(
        full / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "correct": i < 8,
                "truncated": False,
            }
            for i in range(10)
        ],
    )
    _write(
        comp / "math500" / "r.jsonl",
        [
            {
                "pid": f"p{i}",
                "decoding": {"seed": i},
                "press": "random",
                "budget": 256,
                "correct": i < 3,
                "truncated": False,
                "n_tokens": 5,
            }
            for i in range(10)
        ],
    )
    _write(
        readout / "readout" / "r.jsonl",
        [
            {
                "reference_key": ["math500", f"p{i}", "m", "fullcache", "full", i],
                "condition": {"press": "random", "budget": 256},
                "readout_correct": i < 6,
                "generated_token_ids": [1],
            }
            for i in range(10)
        ],
    )
    _, cells = MODULE.build_report(
        fullcache_root=full, compression_root=comp, readout_root=readout
    )
    cell = cells[0]
    assert cell["full_accuracy"] == approx(cell["full_accuracy_all_rows"])
    lo, hi = cell["primary_bounds"]["readout_accuracy"]
    assert lo == approx(hi) == approx(cell["readout_accuracy"])
    lo, hi = cell["primary_bounds"]["residual"]
    assert lo == approx(hi) == approx(cell["unexplained_residual"])

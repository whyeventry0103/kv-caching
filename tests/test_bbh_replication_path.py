"""BBH logical-deduction calibration path (held), CPU only, plus the
capacity-feasibility arithmetic for the replication design.

The BBH timing bank runs the same manifest-bank runner as GSM8K end to end on
stubs (model, generation, the blob-verified file loader); the real row and
sidecar builders, the pair publication path and the timing reader run, and the
reader scores stored traces with the frozen ``bbh_grade``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import nightly_autorun as scheduler  # noqa: E402
import run_bank_manifest as rbm  # noqa: E402

from kvcache import bbh  # noqa: E402
from kvcache.datasets import Problem  # noqa: E402
from kvcache.gsm8k_timing import read_timing, score_rows  # noqa: E402
from kvcache.manifest_capacity import predicted_exposure, prompt_fit  # noqa: E402
from kvcache.manifest_grid import resolve_grid_design  # noqa: E402
from kvcache.manifest_problems import load_frozen_pool, manifest_key  # noqa: E402

CONFIG = ROOT / "configs/experiments/bank_bbh_timing_m5.yaml"
MANIFEST = json.loads((ROOT / "configs/meeting5/problem_manifests.json").read_text())
LDE = MANIFEST["bbh_logical_deduction_five_objects"]
PIDS = LDE["calibration"]
SEEDS = [11, 23]
SPLIT = {"split": "calibration", "revision": LDE["revision"]}
BENCHES = yaml.safe_load((ROOT / "configs/benchmarks.yaml").read_text())["benchmarks"]
GOLDS = {f"bbh_lde5/{i}": f"({'ABCDE'[i % 5]})" for i in range(250)}


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_bbh_benchmark_entry_maps_to_its_frozen_manifest_and_pin():
    cfg = BENCHES["bbh_lde5"]
    assert manifest_key("bbh_lde5", cfg) == "bbh_logical_deduction_five_objects"
    assert manifest_key("gsm8k", BENCHES["gsm8k"]) == "gsm8k"
    assert cfg["revision"] == bbh.PINNED_COMMIT == LDE["revision"]


def test_bbh_timing_config_is_held_and_resolves_the_frozen_calibration_split():
    cfg = _config()
    assert cfg["held"] is True and cfg["scale_authorized"] is False
    d = rbm.resolve_design(cfg)
    assert d["benchmark"] == "bbh_lde5" and d["pids"] == PIDS and len(PIDS) == 25
    assert d["n_pool"] == 250 and d["n_generations"] == 50
    assert d["max_new_tokens"] == 4096 and d["revision"] == bbh.PINNED_COMMIT


def test_frozen_pool_dispatch_and_refusals(monkeypatch, tmp_path):
    calls = []

    def hf(bench, cfg, **k):
        calls.append((bench, k))
        return ["pool"]

    assert load_frozen_pool("gsm8k", BENCHES["gsm8k"], hf_loader=hf) == ["pool"]
    assert calls == [("gsm8k", {"subset": None, "allow_download": False})]
    seen = []
    monkeypatch.setattr(bbh, "load_bbh", lambda p: seen.append(p) or ["bbh"])
    got = load_frozen_pool(
        "bbh_lde5", BENCHES["bbh_lde5"], hf_loader=hf, data_root=tmp_path
    )
    assert got == ["bbh"] and seen == [bbh.data_path(tmp_path)]
    assert len(calls) == 1  # BBH never goes through the HF loader
    with pytest.raises(ValueError, match="not the pinned BBH commit"):
        load_frozen_pool(
            "bbh_lde5", {**BENCHES["bbh_lde5"], "revision": "0" * 40}, hf_loader=hf
        )
    monkeypatch.delenv("REMOTE_DATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="REMOTE_DATA_ROOT"):
        load_frozen_pool("bbh_lde5", BENCHES["bbh_lde5"], hf_loader=hf)


def test_score_rows_dispatches_the_bbh_grader_and_checks_option_golds():
    rows = [
        {"pid": "bbh_lde5/2", "trace": "x</think> \\boxed{(C)}", "truncated": False},
        {"pid": "bbh_lde5/3", "trace": "x</think> \\boxed{(C)}", "truncated": False},
        {"pid": "bbh_lde5/4", "trace": "x</think> \\boxed{(E)}", "truncated": True},
    ]
    got = score_rows(rows, GOLDS, revision=bbh.PINNED_COMMIT, bench="bbh_lde5")
    assert [s["correct"] for s in got["scores"]] == [True, False, False]
    assert got["scores"][2]["rule"] == "truncated"
    with pytest.raises(ValueError, match="is not one option"):
        score_rows(
            rows[:1],
            {"bbh_lde5/2": "C"},
            revision=bbh.PINNED_COMMIT,
            bench="bbh_lde5",
        )
    with pytest.raises(ValueError, match="is not a number"):  # GSM8K unchanged
        score_rows([{**rows[0], "pid": "g"}], {"g": "(C)"}, revision="r", bench="gsm8k")


class _Tok:
    def __init__(self):
        self.texts: dict[tuple[int, ...], str] = {}

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[tuple(ids)]


def _plan(pid: str, seed: int) -> dict:
    """seed 11, i == 0: truncated (gold boxed); seed 23, i < 4: the wrong
    option; everything else names the gold option in its answer region."""
    i = PIDS.index(pid)
    gold = GOLDS[pid]
    wrong = "(A)" if gold != "(A)" else "(B)"
    answer = wrong if seed == 23 and i < 4 else gold
    return {
        "i": i,
        "trace": f"order the objects</think> The answer is {answer}.",
        "truncated": seed == 11 and i == 0,
        "n_tokens": 300 + 20 * i,
        "prompt_len": 140 + i,
    }


def test_bbh_timing_bank_end_to_end_reads_final_with_the_bbh_grader(
    tmp_path, monkeypatch
):
    cfg = {**_config(), "held": False, "scale_authorized": True}
    cfg_path = tmp_path / "bank.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "runs" / "bank-bbh-timing-m5"
    out.mkdir(parents=True)
    tok = _Tok()
    pool = [Problem(p, f"q {p}", g, {}) for p, g in GOLDS.items()]

    def generate(model, tokenizer, problem, *a, seed, **k):
        plan = _plan(problem.pid, seed)
        base = 10_000 * (plan["i"] + 1) + 1_000_000 * seed
        prompt_ids = tuple(base + j for j in range(plan["prompt_len"]))
        trace_ids = tuple(base + 5_000 + j for j in range(plan["n_tokens"]))
        prompt = f"prompt {problem.pid}"
        tok.texts[prompt_ids], tok.texts[trace_ids] = prompt, plan["trace"]
        return SimpleNamespace(
            prompt=prompt,
            trace=plan["trace"],
            answer_raw="x",
            answer_canon="x",
            correct=False,
            n_tokens=plan["n_tokens"],
            truncated=plan["truncated"],
            wallclock_s=0.025 * plan["n_tokens"],
            hit_token_cap=plan["truncated"],
            has_answer_marker=not plan["truncated"],
            finish_reason="length" if plan["truncated"] else "marker",
            prompt_token_ids=prompt_ids,
            trace_token_ids=trace_ids,
        )

    monkeypatch.setattr(rbm, "load_model_and_tokenizer", lambda *a, **k: (None, tok))
    monkeypatch.setattr(rbm, "generate_full_cache", generate)
    real_runmeta = rbm.capture_runmeta
    monkeypatch.setattr(
        rbm,
        "capture_runmeta",
        lambda *a, **k: {**real_runmeta(*a, **k), "source_tree_hash": "tree-test"},
    )
    monkeypatch.setattr(bbh, "load_bbh", lambda path: pool)
    monkeypatch.setattr(
        rbm, "load_benchmark", lambda *a, **k: pytest.fail("BBH used the HF loader")
    )
    monkeypatch.setenv("REMOTE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(out))
    argv = ["run_bank_manifest", "--config", str(cfg_path), "--i-understand"]
    monkeypatch.setattr(sys, "argv", argv)
    assert rbm.main() == 0

    read = read_timing(
        out,
        bench="bbh_lde5",
        pids=PIDS,
        seeds=SEEDS,
        manifest_split=SPLIT,
        golds={p: GOLDS[p] for p in PIDS},
    )
    assert read["status"] == "FINAL" and read["validated_pairs"] == 50
    assert read["score_source"].startswith("bbh_grade")
    assert read["accuracy"] == pytest.approx(45 / 50)
    assert read["truncation_rate"] == pytest.approx(1 / 50)
    assert read["row_correct_field_disagreements"] == 45
    # prompt 140+i plus trace 300+20i always exceeds 256; exceeds 1024 iff
    # 440 + 21i > 1024, i.e. i >= 28 -- never in a 25-problem split
    assert read["budget_would_bind"] == {"256": 1.0, "1024": 0.0}


def test_bbh_pilot_is_blocked_and_its_cap_is_only_proposed(tmp_path):
    _, jobs = scheduler.load_queue(scheduler.QUEUE_DEFAULT)
    job = next(j for j in jobs if j.name == "bank-bbh-timing-m5")
    assert job.blocked and job.campaign is None and job.total == 50
    assert "M5B-BBH-TIMING" not in scheduler.load_campaign_caps()
    eligible, notes = scheduler.launch_candidates(
        tmp_path, [job], skip=set(), live={}, state={}
    )
    assert eligible == []
    assert any("BLOCKED bank-bbh-timing-m5" in s for s in notes)


@pytest.mark.parametrize(
    "alias, launch_id",
    [
        ("bank_manifest_bbh", None),
        ("bank_manifest_bbh", "M5B-GSM8K-TIMING-x"),
        ("bank_manifest", "M5B-BBH-TIMING-x"),
    ],
)
def test_launch_aliases_refuse_the_bbh_config_without_its_own_cap(
    tmp_path, alias, launch_id
):
    from test_m5b_schedule import _launch

    env = dict(os.environ, KVCACHE_REMOTE="1", KVCACHE_MAX_HOURS="1")
    env.pop("KVCACHE_LAUNCH_ID", None)
    if launch_id:
        env["KVCACHE_LAUNCH_ID"] = launch_id
    p = _launch(tmp_path, alias, "configs/experiments/bank_bbh_timing_m5.yaml", env)
    assert p.returncode == 2
    assert "requires its capped scheduler reservation" in p.stderr


def test_grid_design_resolves_bbh_by_its_manifest_key():
    cfg = yaml.safe_load(
        (ROOT / "configs/experiments/grid_gsm8k_template_m5b.yaml").read_text()
    )
    cfg.update(
        benchmarks=["bbh_lde5"],
        max_new_tokens={"bbh_lde5": 4096},
        estimated_generation_count=200 * 2 * len(cfg["presses"]) * len(cfg["budgets"]),
    )
    cfg["manifest"] = {**cfg["manifest"], "split": "confirmation"}
    d = resolve_grid_design(cfg, kind="grid")
    assert d["pids"] == LDE["confirmation"] and d["n_pool"] == 250
    assert d["revision"] == bbh.PINNED_COMMIT and d["max_new_tokens"] == 4096


def test_prompt_fit_by_hand():
    lens = {"a": 50, "b": 192, "c": 193, "d": 300}
    f = prompt_fit(lens, 256)
    # P+64 <= 256 iff P <= 192: a, b fit; c, d relaxed to P+64 (257, 364)
    assert (f["n_strict_fit"], f["n_relaxed"]) == (2, 2)
    assert f["relaxed_pids"] == ["c", "d"]
    assert f["b_eff"]["min"] == 256 and f["b_eff"]["max"] == 364
    # trace slots: a 206, b 64, c 64, d 64; strict-fit only: 206, 64
    assert f["trace_slots_all"]["min"] == 64 and f["trace_slots_all"]["max"] == 206
    assert f["trace_slots_strict_fit"]["median"] == pytest.approx(135)
    with pytest.raises(ValueError):
        prompt_fit({}, 256)


def test_predicted_exposure_by_hand():
    lens = {"a": 50, "c": 193}
    traces = [
        {"pid": "a", "n_tokens": 206},  # 256 = B_eff: never exceeds
        {"pid": "a", "n_tokens": 300},  # 350 > 256: excess 94
        {"pid": "c", "n_tokens": 100},  # B_eff 257 (relaxed): 293 > 257, 36
    ]
    e = predicted_exposure(lens, traces, 256)
    assert e["all"]["n"] == 3 and e["all"]["reached"] == 2
    assert e["all"]["excess_tokens"]["min"] == 36
    assert e["all"]["excess_tokens"]["max"] == 94
    assert e["strict_fit"]["n"] == 2 and e["strict_fit"]["reached"] == 1
    assert e["strict_fit"]["reached_rate"] == pytest.approx(0.5)

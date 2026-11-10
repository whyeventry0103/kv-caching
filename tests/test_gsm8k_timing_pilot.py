"""GSM8K timing pilot (held), CPU only: the manifest-bank runner publishes
row + token-ID pairs through the Meeting 5 recoverable path, an interruption
after the result write resumes to exactly one matching pair per expected key,
and the timing reader validates all 48 keys and scores stored traces with the
frozen GSM8K grader, never the row's ``correct`` field.

The real runner runs end to end on stubs for the model, the dataset pool and
generation; the real row builder, sidecar builder and validators run."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import read_gsm8k_timing as rgt  # noqa: E402
import run_bank_manifest as rbm  # noqa: E402

from kvcache import m5_publish as mp  # noqa: E402
from kvcache.datasets import Problem  # noqa: E402
from kvcache.gsm8k_timing import expected_keys, read_timing  # noqa: E402
from kvcache.manifest_publish import bench_scope, publish_pair  # noqa: E402
from kvcache.token_ids import text_sha256  # noqa: E402

CONFIG = ROOT / "configs/experiments/bank_gsm8k_timing_m5.yaml"
MANIFEST = json.loads((ROOT / "configs/meeting5/problem_manifests.json").read_text())
GSM = MANIFEST["gsm8k"]
PIDS = GSM["calibration"]
SEEDS = [11, 23]
SPLIT = {"split": "calibration", "revision": GSM["revision"]}
RESOLUTION = json.loads(
    (ROOT / "results/diag/meeting5/gsm8k_timing/split_resolution.json").read_text()
)
GOLDS = {p["pid"]: p["gold"] for p in RESOLUTION["problems"]}


class Interrupt(Exception):
    pass


class _Tok:
    """Decodes exactly the IDs the generation stub registered."""

    def __init__(self):
        self.texts: dict[tuple[int, ...], str] = {}

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[tuple(ids)]


def _plan(pid: str, seed: int) -> dict:
    """Hand-fixed outcome per condition: index i in the split, two seeds.

    seed 11, i == 0: hits the cap without a marker (truncated, boxed gold);
    seed 23, i < 4: boxes gold + 1 (wrong); everything else boxes the gold.
    The generation-time ``correct`` field is always False."""
    i = PIDS.index(pid)
    truncated = seed == 11 and i == 0
    value = int(GOLDS[pid]) + (1 if seed == 23 and i < 4 else 0)
    return {
        "i": i,
        "trace": f"work for {pid} s{seed}</think> The total is \\boxed{{{value}}}.",
        "truncated": truncated,
        "n_tokens": 100 + 10 * i,
        "prompt_len": 120 + i,
    }


def _setup(tmp_path, monkeypatch):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(held=False, scale_authorized=True)
    cfg_path = tmp_path / "bank.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "runs" / "bank-gsm8k-timing-m5"
    out.mkdir(parents=True)
    tok, n = _Tok(), {"generations": 0}
    pool = [
        Problem(p, f"q {p}", GOLDS.get(p, "0"), {})
        for p in GSM["calibration"] + GSM["confirmation"] + GSM["reserve"]
    ]

    def generate(model, tokenizer, problem, *a, seed, **k):
        n["generations"] += 1
        plan = _plan(problem.pid, seed)
        base = 10_000 * (plan["i"] + 1) + 1_000_000 * seed + 7 * n["generations"]
        prompt_ids = tuple(base + j for j in range(plan["prompt_len"]))
        trace_ids = tuple(base + 500 + j for j in range(plan["n_tokens"]))
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
    real_runmeta = rbm.capture_runmeta  # on the host the sync state supplies it
    monkeypatch.setattr(
        rbm,
        "capture_runmeta",
        lambda *a, **k: {**real_runmeta(*a, **k), "source_tree_hash": "tree-test"},
    )
    monkeypatch.setattr(rbm, "load_benchmark", lambda *a, **k: pool)
    monkeypatch.setenv("KVCACHE_REMOTE", "1")
    monkeypatch.setenv("REMOTE_RUNS_ROOT", str(out))
    return cfg_path, out, n


def _run(monkeypatch, cfg_path, *extra):
    argv = ["run_bank_manifest", "--config", str(cfg_path), "--i-understand", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return rbm.main()


def _lines(out, sub):
    return [
        json.loads(x)
        for p in sorted((out / sub).rglob("*.jsonl"))
        for x in p.read_text().splitlines()
        if x.strip()
    ]


def _read(out):
    return read_timing(
        out, bench="gsm8k", pids=PIDS, seeds=SEEDS, manifest_split=SPLIT, golds=GOLDS
    )


def _assert_exactly_one_pair_per_key(out):
    rows, sides = _lines(out, "gsm8k"), _lines(out, "token_ids/gsm8k")
    want = expected_keys(PIDS, SEEDS)
    assert len(want) == 48
    assert sorted(map(mp.row_key, rows)) == sorted(want)
    assert sorted(map(mp.row_key, sides)) == sorted(want)
    by = {mp.row_key(s): s for s in sides}
    for r in rows:
        s = by[mp.row_key(r)]
        assert s["trace_text_sha256"] == text_sha256(r["trace"])
        assert len(s["trace_token_ids"]) == r["n_tokens"]
        assert r["manifest_split"] == s["manifest_split"] == SPLIT


@pytest.mark.parametrize("journal", ["kept", "lost"])
def test_interrupt_after_result_write_resumes_to_one_pair_per_key(
    tmp_path, monkeypatch, journal
):
    cfg_path, out, n = _setup(tmp_path, monkeypatch)
    real, state = mp._publish_one, {"fired": False}

    def stop_before_sidecar(path, rec):
        if not state["fired"] and "token_ids" in path.parts:
            state["fired"] = True
            raise Interrupt("stopped after the result write")
        return real(path, rec)

    monkeypatch.setattr(mp, "_publish_one", stop_before_sidecar)
    with pytest.raises(Interrupt):
        _run(monkeypatch, cfg_path)
    # The stop left one published row, no sidecar, and the envelope pending.
    assert len(_lines(out, "gsm8k")) == 1
    assert _lines(out, "token_ids/gsm8k") == []
    pending = [
        p for p in (out / mp.JOURNAL).rglob("*.json") if ".published" not in p.name
    ]
    assert len(pending) == 1
    assert mp.BENCH == "math500"  # the scope is restored even on a stop
    if journal == "lost":  # the legacy failure: a row orphan with no envelope
        pending[0].unlink()

    assert _run(monkeypatch, cfg_path, "--resume") == 0
    _assert_exactly_one_pair_per_key(out)
    # kept: the envelope is published, nothing is regenerated;
    # lost: the orphan row is quarantined (kept as evidence) and regenerated.
    assert n["generations"] == (48 if journal == "kept" else 49)
    read = _read(out)
    assert read["status"] == "FINAL" and read["validated_pairs"] == 48
    assert read["quarantined_records"] == (0 if journal == "kept" else 1)
    assert not (out / "math500").exists() and mp.BENCH == "math500"

    # A further resume is a no-op: complete pairs are skipped.
    assert _run(monkeypatch, cfg_path, "--resume") == 0
    assert n["generations"] == (48 if journal == "kept" else 49)
    _assert_exactly_one_pair_per_key(out)


@pytest.fixture
def complete_run(tmp_path, monkeypatch):
    cfg_path, out, _ = _setup(tmp_path, monkeypatch)
    assert _run(monkeypatch, cfg_path) == 0
    return out


def test_reader_scores_stored_traces_with_the_frozen_grader(complete_run):
    read = _read(complete_run)
    # 48 conditions: 1 truncated (boxed gold, still incorrect), 4 wrong boxes.
    assert read["accuracy"] == pytest.approx(43 / 48)
    # i == 0 wrong on both seeds; i = 1..3 right on seed 11 only; 20 all right
    assert read["accuracy_equal_problem"] == pytest.approx((20 + 3 * 0.5) / 24)
    assert read["grader"] == "gsm8k-numeric-v2"
    assert read["grader_rules"] == {"boxed": 47, "truncated": 1}
    # every row says correct=False; the read never uses that field as a score
    assert read["row_correct_field_disagreements"] == 43
    assert read["truncation_rate"] == pytest.approx(1 / 48)
    lengths = [100 + 10 * i for i in range(24)]
    assert read["trace_tokens"]["total"] == 2 * sum(lengths)
    assert read["trace_tokens"]["median"] == pytest.approx(215.0)
    assert read["seconds_per_token"] == pytest.approx(0.025)
    # prompt 120+i plus trace 100+10i exceeds 256 iff i >= 4; never 1024
    assert read["budget_would_bind"] == {"256": 40 / 48, "1024": 0.0}


def _drop_first(path: Path) -> None:
    lines = path.read_text().splitlines()
    path.write_text("".join(x + "\n" for x in lines[1:]))


def _dup_first(path: Path) -> None:
    lines = path.read_text().splitlines()
    path.write_text("".join(x + "\n" for x in [*lines, lines[0]]))


def _edit_first(path: Path, **over) -> None:
    lines = path.read_text().splitlines()
    rec = {**json.loads(lines[0]), **over}
    path.write_text("".join(x + "\n" for x in [json.dumps(rec), *lines[1:]]))


def _extra(path: Path) -> None:
    rec = json.loads(path.read_text().splitlines()[0])
    rec["pid"] = GSM["confirmation"][0]
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


ROWS = "gsm8k"
SIDES = "token_ids/gsm8k"


@pytest.mark.parametrize(
    "sub, damage, match",
    [
        (SIDES, _drop_first, "lack sidecars"),
        (ROWS, _drop_first, "lack result rows"),
        (ROWS, _dup_first, "repeated result rows"),
        (SIDES, _dup_first, "repeated sidecars"),
        (ROWS, _extra, "outside the design"),
        (ROWS, lambda p: _edit_first(p, trace="edited"), "trace hash"),
        (ROWS, lambda p: _edit_first(p, manifest_split={}), "manifest_split"),
        (ROWS, lambda p: p.write_text(p.read_text() + "{torn\n"), "torn"),
    ],
)
def test_reader_refuses_anything_but_48_matching_pairs(
    complete_run, sub, damage, match
):
    damage(sorted((complete_run / sub).rglob("*.jsonl"))[0])
    with pytest.raises(ValueError, match=match):
        _read(complete_run)


def test_reader_refuses_a_pending_envelope_and_a_missing_gold(complete_run):
    pend = complete_run / mp.JOURNAL / "fullcache-full-s11" / "x.json"
    pend.write_text("{}")
    with pytest.raises(ValueError, match="pending"):
        _read(complete_run)
    pend.unlink()
    with pytest.raises(ValueError, match="no gold"):
        read_timing(
            complete_run,
            bench="gsm8k",
            pids=PIDS,
            seeds=SEEDS,
            manifest_split=SPLIT,
            golds={k: v for k, v in GOLDS.items() if k != PIDS[0]},
        )


def test_recovery_refuses_a_pending_envelope_from_another_design(
    tmp_path, monkeypatch, complete_run
):
    src = complete_run
    (tmp_path / "second").mkdir()
    cfg_path, out, _ = _setup(tmp_path / "second", monkeypatch)
    row = _lines(src, ROWS)[0]
    side = _lines(src, SIDES)[0]
    row["manifest_split"] = side["manifest_split"] = {"split": "confirmation"}
    mp.write_journal(out, row, side)
    with pytest.raises(SystemExit, match="REFUSED: pending envelope"):
        _run(monkeypatch, cfg_path, "--resume")


def test_bench_scope_restores_and_refuses_misfiled_records(tmp_path):
    with pytest.raises(RuntimeError, match="boom"), bench_scope("gsm8k"):
        assert mp.BENCH == "gsm8k"
        raise RuntimeError("boom")
    assert mp.BENCH == "math500"
    with bench_scope("gsm8k"), pytest.raises(RuntimeError, match="nested"):
        with bench_scope("bbh"):
            pass
    assert mp.BENCH == "math500"
    rec = {"bench": "math500", "pid": "p", "press": "fullcache", "budget": "full"}
    with pytest.raises(ValueError, match="not 'gsm8k'"):
        publish_pair(tmp_path, "gsm8k", {**rec, "seed": 11}, {**rec, "seed": 11})


def test_runner_refuses_a_bank_without_token_sidecars():
    cfg = {**yaml.safe_load(CONFIG.read_text()), "store_token_ids": False}
    with pytest.raises(ValueError, match="pairs"):
        rbm.resolve_design(cfg)


def test_read_script_writes_a_final_read_and_retires_on_refusal(
    tmp_path, monkeypatch, complete_run
):
    monkeypatch.setattr(rgt, "dataset_golds", lambda design, committed: GOLDS)
    out = tmp_path / "timing_read.json"
    argv = ["read", "--runs-root", str(complete_run), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert rgt.main() == 0
    read = json.loads(out.read_text())
    assert read["status"] == "FINAL" and read["validated_pairs"] == 48
    shutil.rmtree(complete_run / SIDES)
    with pytest.raises(SystemExit, match="REFUSED"):
        rgt.main()
    assert not out.exists()  # the stale FINAL is retired, not left behind
    assert list(tmp_path.glob("timing_read.json.superseded-*"))


def test_read_script_refuses_cleanly_without_the_resolution_record(
    tmp_path, monkeypatch
):
    # results/ is not source-synced: a host without the record must refuse
    # with the fix named, not trace back.
    argv = ["read", "--dry-run", "--resolution", str(tmp_path / "absent.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="REFUSED: resolution record .* missing"):
        rgt.main()

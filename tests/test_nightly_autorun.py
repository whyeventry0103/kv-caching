import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nightly_autorun", ROOT / "scripts" / "nightly_autorun.py"
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Register before exec so module-level @dataclass resolves its namespace (py3.14).
sys.modules["nightly_autorun"] = M
SPEC.loader.exec_module(M)


def _job(name, run_name, total, kind, requires=None, seeds=(11, 23, 47, 89)):
    return M.Job(
        name=name,
        command="grid",
        config="c.yaml",
        run_name=run_name,
        total=total,
        seeds=list(seeds),
        min_free_gb=24,
        progress_kind=kind,
        requires=requires,
    )


def _gpu(idx, free_mb):
    return M.GpuState(idx, 49140, free_mb, 0)


# ---- nvidia-smi parsing ----


def test_parse_nvidia_smi():
    text = "0, 49140, 47000, 5\n1, 49140, 12000, 88\n"
    gpus = M.parse_nvidia_smi(text)
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].mem_free_mb == 47000 and gpus[1].util_pct == 88


# ---- eligibility + sharding ----


def test_eligibility_skips_full_gpu():
    gpus = [_gpu(0, 47000), _gpu(1, 12000)]  # 45.9GB vs 11.7GB
    elig = M.eligible_gpus(gpus, min_free_gb=24)
    assert [g.index for g in elig] == [0]


def test_eligibility_sorts_by_free_desc():
    gpus = [_gpu(0, 40000), _gpu(1, 46000)]
    assert [g.index for g in M.eligible_gpus(gpus, 24)] == [1, 0]


def test_shard_two_gpus_even_split():
    gpus = [_gpu(1, 46000), _gpu(0, 40000)]  # freest first
    assert M.shard_seeds([11, 23, 47, 89], gpus) == {1: [11, 47], 0: [23, 89]}


def test_shard_one_gpu_all_seeds():
    assert M.shard_seeds([11, 23, 47, 89], [_gpu(0, 47000)]) == {0: [11, 23, 47, 89]}


# ---- max-hours ----


def test_max_hours_uses_full_window():
    # 2-decimal floor: near-full window, never overshoots 09:00 (int floor lost ~1h).
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 0)) == 7.0
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 0, 1)) == 6.99  # 1s late != 6
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 40)) == 6.33
    assert M.compute_max_hours(datetime(2026, 6, 24, 3, 20)) == 5.66


def test_max_hours_sub_one_hour_is_none():
    assert M.compute_max_hours(datetime(2026, 6, 24, 8, 30)) is None  # 0.5h -> skip
    assert M.compute_max_hours(datetime(2026, 6, 24, 9, 30)) is None  # past stop


# ---- progress counting ----


def _write(path: Path, n_rows: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(n_rows)))


def test_count_progress_grid_excludes_token_ids(tmp_path):
    job = _job("g", "g", 800, "grid")
    _write(tmp_path / "g" / "math500" / "m" / "a-256-s11.jsonl", 100)
    _write(tmp_path / "g" / "token_ids" / "math500" / "m" / "a-256-s11.jsonl", 100)
    assert M.count_progress(tmp_path, job) == 100  # sidecars not double-counted


def test_count_progress_readout(tmp_path):
    job = _job("r", "r", 800, "readout")
    _write(tmp_path / "r" / "readout" / "fullcache-256-s11.jsonl", 42)
    assert M.count_progress(tmp_path, job) == 42


def test_count_progress_missing_dir_is_zero(tmp_path):
    assert M.count_progress(tmp_path, _job("x", "x", 800, "grid")) == 0


def test_count_per_seed_sums_over_presses(tmp_path):
    # a seed's count sums its per-press files; recovery uses this to find dead shards
    job = _job("g", "g", 800, "grid", seeds=(11, 23, 47, 89))
    _write(tmp_path / "g" / "math500" / "m" / "streamingllm-256-s11.jsonl", 100)
    _write(tmp_path / "g" / "math500" / "m" / "random-256-s11.jsonl", 100)
    _write(tmp_path / "g" / "math500" / "m" / "streamingllm-256-s23.jsonl", 40)
    per = M.count_per_seed(tmp_path, job)
    assert per[11] == 200 and per[23] == 40  # seed 11 complete (200), seed 23 partial
    # seed_total = 800//4 = 200 -> incomplete seeds are everything except 11
    seed_total = job.total // len(job.seeds)
    incomplete = [s for s in job.seeds if per.get(s, 0) < seed_total]
    assert incomplete == [23, 47, 89]


# ---- job selection + dependency gating ----


def test_select_first_incomplete(tmp_path):
    jobs = [_job("a", "a", 800, "grid")]
    job, reason = M.select_job(tmp_path, jobs)
    assert job is not None and job.name == "a" and reason == "selected"


def test_select_blocks_on_incomplete_dependency(tmp_path):
    jobs = [
        _job("grid", "grid", 800, "grid"),
        _job("rev", "rev", 800, "readout", requires="grid"),
    ]
    _write(tmp_path / "grid" / "math500" / "m" / "a-256-s11.jsonl", 300)  # 300/800
    job, reason = M.select_job(tmp_path, jobs)
    assert job is not None and job.name == "grid"  # rev blocked, grid selected


def test_select_unblocks_when_dependency_complete(tmp_path):
    jobs = [
        _job("grid", "grid", 800, "grid"),
        _job("rev", "rev", 800, "readout", requires="grid"),
    ]
    _write(tmp_path / "grid" / "math500" / "m" / "a-256-s11.jsonl", 800)  # complete
    job, reason = M.select_job(tmp_path, jobs)
    assert job is not None and job.name == "rev"


def test_select_skips_stuck_job(tmp_path):
    # a stuck job (repeated no-progress failures) is skipped, queue advances to the next
    jobs = [_job("a", "a", 800, "grid"), _job("b", "b", 800, "grid")]
    job, reason = M.select_job(tmp_path, jobs, skip={"a"})
    assert job is not None and job.name == "b"


def test_select_all_complete(tmp_path):
    jobs = [_job("a", "a", 800, "grid")]
    _write(tmp_path / "a" / "math500" / "m" / "a-256-s11.jsonl", 800)
    job, reason = M.select_job(tmp_path, jobs)
    assert job is None and reason == "all jobs complete"


# ---- watcher: our-sessions detection (gate against collide/double-launch) ----


def test_our_sessions_running_matches_our_naming(monkeypatch):
    out = (
        "grid-diss256-tok-g0: 1 windows (created ...)\n"
        "grid-diss256-tok-g1: 1 windows (created ...)\n"
        "someone-else: 1 windows\n"
    )

    class R:
        returncode = 0
        stdout = out

    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: R())
    running = M.our_sessions_running(["grid-diss256-tok", "reverse-readout-diss256"])
    # the other user's session is excluded
    assert running == ["grid-diss256-tok-g0", "grid-diss256-tok-g1"]


def test_our_sessions_running_no_server(monkeypatch):
    class R:
        returncode = 1  # "no server running on ..."
        stdout = ""

    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: R())
    assert M.our_sessions_running(["grid-diss256-tok"]) == []


# ---- launch spec wiring ----


def test_launch_spec_env_and_argv(tmp_path):
    job = _job("grid-diss256-tok", "grid-diss256-tok", 800, "grid")
    specs = M.build_launch_specs(job, {0: [11, 23], 1: [47, 89]}, max_hours=7)
    assert {s.gpu_index for s in specs} == {0, 1}
    s0 = next(s for s in specs if s.gpu_index == 0)
    assert s0.env["CUDA_VISIBLE_DEVICES"] == "0"
    assert s0.env["KVCACHE_RUN_NAME"] == "grid-diss256-tok"
    assert s0.env["KVCACHE_MAX_HOURS"] == "7"
    assert s0.env["KVCACHE_ALLOW_EXISTING_RUN"] == "1"
    assert s0.launch_argv[:3] == ["bash", "scripts/launch.sh", "--i-understand"]
    assert s0.launch_argv[3:5] == ["grid", "c.yaml"]
    assert s0.launch_argv[-3:] == ["--resume", "--seeds", "11,23"]

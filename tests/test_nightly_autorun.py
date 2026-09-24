import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

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


def test_shard_fair_share_holds_seeds_for_busy_gpus():
    # Regression (2026-07-19): one free GPU of two grabbed ALL four readout
    # seeds while the other GPU was finishing a grid tail; the second GPU then
    # idled for the rest of the window. With total_gpus=2 the lone free GPU
    # takes only its stride share; the rest stay needy for the next tick.
    assignment = M.shard_seeds([11, 23, 47, 89], [_gpu(0, 47000)], total_gpus=2)
    assert assignment == {0: [11, 47]}
    # Both free out of two: unchanged even split.
    gpus = [_gpu(1, 46000), _gpu(0, 40000)]
    assert M.shard_seeds([11, 23, 47, 89], gpus, total_gpus=2) == {
        1: [11, 47],
        0: [23, 89],
    }


# ---- max-hours ----


def test_max_hours_uses_full_window():
    # 2-decimal floor: near-full window, never overshoots the 10:45 stop.
    assert M.compute_max_hours(datetime(2026, 6, 24, 1, 45)) == 9.0  # start -> full 9h
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 0)) == 8.75
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 0, 1)) == 8.74  # 1s late
    assert M.compute_max_hours(datetime(2026, 6, 24, 2, 40)) == 8.08
    assert M.compute_max_hours(datetime(2026, 6, 24, 3, 20)) == 7.41


def test_max_hours_sub_one_hour_is_none():
    assert M.compute_max_hours(datetime(2026, 6, 24, 9, 45)) == 1.0  # 1h -> runs
    assert M.compute_max_hours(datetime(2026, 6, 24, 9, 50)) is None  # 0.92h -> skip
    assert M.compute_max_hours(datetime(2026, 6, 24, 10, 0)) is None  # 0.75h -> skip
    assert M.compute_max_hours(datetime(2026, 6, 24, 11, 0)) is None  # past stop


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


# ---- multi-job GPU fill (launch_candidates + live_map) ----


def test_live_map_one_parse_matches_exact_run_names():
    live = M.live_map(
        ["grid-a", "grid-a-x"],
        ["grid-a-g0", "grid-a-x-g1", "other-g0", "grid-a-g1extra"],
    )
    # exact `<run>-g<idx>` only: no prefix bleed between grid-a and grid-a-x,
    # and a trailing-suffix session name does not count.
    assert live == {"grid-a": {0}, "grid-a-x": {1}}


def test_candidates_skip_all_in_flight_and_yield_next_job(tmp_path):
    # job a: seeds 11,23 remain, BOTH owned by the live gpu0 shard -> skipped with a
    # note; job b (independent) becomes the candidate so gpu1 doesn't idle for hours.
    jobs = [_job("a", "a", 800, "grid", seeds=(11, 23)), _job("b", "b", 800, "grid")]
    _write(tmp_path / "a" / "math500" / "m" / "x-256-s11.jsonl", 100)  # incomplete
    state = {"a": {"shards": {"0": [11, 23]}}}
    live = {"a": {0}, "b": set()}
    candidates, notes = M.launch_candidates(
        tmp_path, jobs, skip=set(), live=live, state=state
    )
    assert [(j.name, seeds) for j, seeds in candidates] == [("b", [11, 23, 47, 89])]
    assert any("a: incomplete seeds all in flight" in n for n in notes)


def test_candidates_partial_shard_death_relaunches_only_dead_seeds(tmp_path):
    # gpu0 shard (seeds 11,23) died (not live); gpu1 shard (47,89) still running ->
    # only the dead shard's incomplete seeds are needy, and job stays first in line.
    jobs = [_job("a", "a", 800, "grid")]
    _write(tmp_path / "a" / "math500" / "m" / "x-256-s11.jsonl", 200)  # 11 complete
    state = {"a": {"shards": {"0": [11, 23], "1": [47, 89]}}}
    live = {"a": {1}}
    candidates, _ = M.launch_candidates(
        tmp_path, jobs, skip=set(), live=live, state=state
    )
    assert [(j.name, seeds) for j, seeds in candidates] == [("a", [23])]


def test_candidates_respect_dependency_and_completion(tmp_path):
    jobs = [
        _job("grid", "grid", 800, "grid"),
        _job("rev", "rev", 800, "readout", requires="grid"),
        _job("later", "later", 400, "grid"),
    ]
    _write(tmp_path / "grid" / "math500" / "m" / "a-256-s11.jsonl", 800)  # complete
    candidates, _ = M.launch_candidates(tmp_path, jobs, skip=set(), live={}, state={})
    # grid complete -> not a candidate; rev unblocked and first; later follows.
    assert [j.name for j, _ in candidates] == ["rev", "later"]


# ---- marker jobs, bench subtrees, extra args ----


def _job2(**kw):
    base = dict(
        name="j",
        command="grid",
        config="c.yaml",
        run_name="j",
        total=400,
        seeds=[11, 23, 47, 89],
        min_free_gb=24,
        progress_kind="grid",
        requires=None,
    )
    base.update(kw)
    return M.Job(**base)


def test_marker_job_progress_and_per_seed(tmp_path):
    job = _job2(
        name="rkv-smoke",
        command="rkvsmoke",
        run_name="rkv-smoke",
        total=1,
        seeds=[11],
        progress_kind="marker",
        marker="rkv-gen-smoke-PASS.json",
    )
    assert M.count_progress(tmp_path, job) == 0
    assert M.count_per_seed(tmp_path, job) == {}
    (tmp_path / "rkv-gen-smoke-PASS.json").write_text("{}")
    assert M.count_progress(tmp_path, job) == 1
    assert M.count_per_seed(tmp_path, job) == {11: 1}


def test_count_progress_bench_subtree(tmp_path):
    job = _job2(name="fc-aime24", run_name="fc-aime24", total=240, bench="aime24")
    _write(tmp_path / "fc-aime24" / "aime24" / "m" / "fullcache-full-s11.jsonl", 30)
    _write(tmp_path / "fc-aime24" / "math500" / "m" / "x-s11.jsonl", 99)  # ignored
    assert M.count_progress(tmp_path, job) == 30
    assert M.count_per_seed(tmp_path, job) == {11: 30}


def test_extra_args_appended_to_launch_argv(tmp_path):
    job = _job2(
        name="fc-llama50",
        command="bank",
        run_name="fc-llama50",
        total=200,
        seeds=[11, 23],
        min_free_gb=30,
        extra_args=("--allow-model-download",),
    )
    specs = M.build_launch_specs(job, {0: [11, 23]}, 5.0)
    assert specs[0].launch_argv[-1] == "--allow-model-download"


def test_a_completed_job_is_never_reported_stuck(tmp_path):
    """Completeness is ground truth and must outrank the failure counter.

    shape-p2v2-restricted failed three launches on 2026-08-25 (its runner
    rejected the orchestrator's --resume/--seeds), was marked stuck, and was
    then completed by hand on 2026-08-27 once the runner was fixed. The
    orchestrator kept logging it as stuck while its artifact sat on disk -- the
    log contradicting the filesystem.
    """

    def job(name):
        return M.Job(
            name=name,
            command="shape",
            config="configs/experiments/x.yaml",
            run_name=name,
            total=1,
            seeds=[11],
            min_free_gb=1,
            progress_kind="marker",
            marker=f"{name}/artifact.json",
            requires=None,
        )

    runs = tmp_path / "runs"
    (runs / "done-job").mkdir(parents=True)
    (runs / "done-job" / "artifact.json").write_text("{}", encoding="utf-8")
    (runs / "really-stuck").mkdir(parents=True)  # no artifact -> incomplete

    jobs = [job("done-job"), job("really-stuck")]
    state = {
        "done-job": {"fails": M.MAX_FAILS},
        "really-stuck": {"fails": M.MAX_FAILS},
    }

    stuck = M.stuck_jobs(runs, jobs, state)
    assert (
        "done-job" not in stuck
    ), "a job whose artifact exists must not be reported stuck"
    assert (
        "really-stuck" in stuck
    ), "a job that really did fail and is incomplete must still be reported"


# --------------------------------------------------------------------------
# Amendment Q storage guard. min_free_gb is a VRAM threshold and says nothing
# about the personal disk budget; 68 of ~70 GB is already spent, so a cell that
# overruns fills a SHARED filesystem and the first symptom is another user's
# job dying.


def _q_job(est_gb=0.0):
    from nightly_autorun import Job

    return Job(
        name="q-cell",
        command="replay",
        config="configs/experiments/replay_sawtooth_i16_q1.yaml",
        run_name="q-cell",
        total=200,
        seeds=[11],
        min_free_gb=24,
        progress_kind="replay",
        requires=None,
        est_output_gb=est_gb,
    )


def test_disk_headroom_is_budget_minus_used():
    from nightly_autorun import disk_headroom_gb

    assert disk_headroom_gb(68.0, 70.0) == pytest.approx(2.0)


def test_job_with_no_estimate_is_allowed_through():
    """The drained historical queue declares none; inventing them would be fiction."""
    from nightly_autorun import check_disk_budget

    check_disk_budget(_q_job(0.0), used_gb=69.9, budget_gb=70.0)


def test_a_cell_that_fits_within_the_margin_launches():
    from nightly_autorun import check_disk_budget

    # 10 GB headroom, cell wants 5, margin 2 -> 7 <= 10, fine
    check_disk_budget(_q_job(5.0), used_gb=60.0, budget_gb=70.0)


def test_a_cell_that_would_exhaust_the_budget_is_refused_loudly():
    from nightly_autorun import check_disk_budget

    with pytest.raises(SystemExit, match="REFUSING TO LAUNCH"):
        check_disk_budget(_q_job(5.0), used_gb=68.0, budget_gb=70.0)


def test_the_safety_margin_is_held_back_not_spent():
    """A cell that exactly fills the headroom still refuses."""
    from nightly_autorun import check_disk_budget

    with pytest.raises(SystemExit, match="margin"):
        check_disk_budget(_q_job(2.0), used_gb=68.0, budget_gb=70.0)


def test_refusal_names_the_reclaim_rule_and_forbids_raising_the_estimate():
    from nightly_autorun import check_disk_budget

    with pytest.raises(SystemExit) as e:
        check_disk_budget(_q_job(50.0), used_gb=68.0, budget_gb=70.0)
    msg = str(e.value)
    assert "never from results/" in msg
    assert "Do not raise est_output_gb" in msg


# --------------------------------------------------------------------------
# A queued job whose progress_kind the orchestrator does not recognise raises
# at launch_candidates, i.e. BEFORE any cell runs and after the window has
# opened -- so the whole night is lost and every job behind it is blocked. This
# actually happened when the Amendment Q cells were first queued with
# progress_kind "replay", which is not one of the three kinds count_progress
# implements. Caught by a remote dry-run; this test moves the catch into the
# local gate, which is where it belongs.


def test_every_queued_progress_kind_is_one_the_orchestrator_implements():
    import yaml

    queue = yaml.safe_load(
        (ROOT / "configs" / "nightly_queue.yaml").read_text(encoding="utf-8")
    )
    known = {"grid", "readout", "marker", "postprocess"}
    bad = {
        j["name"]: j["progress_kind"]
        for j in queue["jobs"]
        if j["progress_kind"] not in known
    }
    assert not bad, (
        f"queued jobs use progress_kind values count_progress does not implement: "
        f"{bad}. Known kinds are {sorted(known)}. A job with an unknown kind "
        f"raises in launch_candidates before anything launches and blocks every "
        f"job behind it."
    )


def test_marker_jobs_declare_a_marker_and_others_do_not_need_one():
    import yaml

    queue = yaml.safe_load(
        (ROOT / "configs" / "nightly_queue.yaml").read_text(encoding="utf-8")
    )
    missing = [
        j["name"]
        for j in queue["jobs"]
        if j["progress_kind"] == "marker" and not j.get("marker")
    ]
    assert not missing, f"marker jobs with no marker path: {missing}"


def test_every_queued_requires_names_a_job_that_exists():
    """A typo in `requires` silently blocks a cell forever."""
    import yaml

    queue = yaml.safe_load(
        (ROOT / "configs" / "nightly_queue.yaml").read_text(encoding="utf-8")
    )
    names = {j["name"] for j in queue["jobs"]}
    dangling = {
        j["name"]: j["requires"]
        for j in queue["jobs"]
        if j.get("requires") and j["requires"] not in names
    }
    assert (
        not dangling
    ), f"jobs depending on names that are not in the queue: {dangling}"


def test_every_queued_config_file_exists():
    import yaml

    queue = yaml.safe_load(
        (ROOT / "configs" / "nightly_queue.yaml").read_text(encoding="utf-8")
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    from postprocess_meeting3 import STAGES

    missing = [
        j["name"]
        for j in queue["jobs"]
        if not j.get("cpu_only") and not (ROOT / j["config"]).exists()
    ]
    assert not missing, f"queued jobs whose config file is absent: {missing}"
    # CPU stages name a postprocessing stage instead of a config file; it must
    # exist, and every config its reader reads must exist too.
    stages = [j for j in queue["jobs"] if j.get("cpu_only")]
    unknown = [j["name"] for j in stages if j["config"] not in STAGES]
    assert not unknown, f"cpu stages naming no postprocessing stage: {unknown}"
    for j in stages:
        args = STAGES[j["config"]]["reader"]
        cfgs = [a for a in args if a.startswith("configs/")]
        assert all((ROOT / c).exists() for c in cfgs), (j["name"], cfgs)
        assert j.get("progress_kind") == "postprocess" and j.get("marker"), j["name"]


def test_every_requires_all_names_a_job_that_exists():
    import yaml

    queue = yaml.safe_load(
        (ROOT / "configs" / "nightly_queue.yaml").read_text(encoding="utf-8")
    )
    names = {j["name"] for j in queue["jobs"]}
    dangling = {
        j["name"]: [r for r in j.get("requires_all") or [] if r not in names]
        for j in queue["jobs"]
    }
    assert not {k: v for k, v in dangling.items() if v}, dangling


def test_storage_is_measured_in_the_same_units_the_budget_is_declared_in(tmp_path):
    """GiB, matching `du -h` and every usage figure in the project's docs.

    Measuring in decimal GB against a GiB budget reported 73.01 against a budget
    of 70 and refused every job unconditionally. A guard that is always on is
    indistinguishable from a broken one, and this is how that was caught.

    The tolerance is deliberately tight enough to separate the two units: GiB
    and decimal GB differ by 7.4%, so a 2% band cannot pass both. The file is
    large enough that `du`'s block rounding is well inside that band.
    """
    from nightly_autorun import storage_used_gb

    n_bytes = 64 * 1024 * 1024
    (tmp_path / "blob.bin").write_bytes(b"\0" * n_bytes)

    used = storage_used_gb(tmp_path)
    gib = n_bytes / (1024**3)
    decimal_gb = n_bytes / 1e9

    assert used == pytest.approx(
        gib, rel=0.02
    ), f"expected GiB ({gib:.6f}); decimal GB would be {decimal_gb:.6f}"
    assert used != pytest.approx(decimal_gb, rel=0.02)


# --------------------------------------------------------------------------
# The guard must charge a cell for what it will still WRITE, not for its total.
# replay-onset-t2048 was refused at 347/400 over a 0.20 GB estimate whose real
# remainder was about 0.03 GB: a nearly finished run blocked over space it was
# never going to use. An unstarted cell must be unaffected, or the fix would
# quietly weaken the guard for every cell that has not run.


def test_unstarted_cell_is_charged_its_full_estimate():
    from nightly_autorun import remaining_output_gb

    assert remaining_output_gb(_q_job(0.20), done=0) == pytest.approx(0.20)


def test_a_nearly_finished_cell_is_charged_only_its_remainder():
    from nightly_autorun import remaining_output_gb

    job = _q_job(0.20)
    object.__setattr__(job, "total", 400)
    # 347/400 done -> 13.25% left
    assert remaining_output_gb(job, done=347) == pytest.approx(0.20 * 53 / 400)


def test_a_complete_cell_projects_nothing():
    from nightly_autorun import remaining_output_gb

    job = _q_job(0.20)
    object.__setattr__(job, "total", 400)
    assert remaining_output_gb(job, done=400) == pytest.approx(0.0)
    assert remaining_output_gb(job, done=999) == pytest.approx(0.0)


def test_the_real_t2048_case_now_launches_and_would_not_have_before():
    from nightly_autorun import DISK_SAFETY_MARGIN_GB, check_disk_budget

    job = _q_job(0.20)
    object.__setattr__(job, "total", 400)
    used, budget = 68.83, 70.0
    headroom = budget - used  # 1.17
    # before: full estimate + margin exceeded headroom
    assert 0.20 + DISK_SAFETY_MARGIN_GB > headroom
    # after: the remainder fits
    check_disk_budget(job, used, budget, done=347)


def test_an_unstarted_cell_that_does_not_fit_is_still_refused():
    """The fix must not make the guard permissive for fresh cells."""
    from nightly_autorun import check_disk_budget

    job = _q_job(0.45)
    object.__setattr__(job, "total", 800)
    with pytest.raises(SystemExit, match="REFUSING TO LAUNCH"):
        check_disk_budget(job, used_gb=68.83, budget_gb=70.0, done=0)


# --------------------------------------------------------------------------
# The nightly report must use the orchestrator's stuck filter, not a raw
# fails>=MAX_FAILS scan. shape-p2v2-restricted failed three launches and was
# later completed by other means; because the report lacked the filter, every
# nightly email was flagged ATTENTION for a job that had finished weeks ago.
# A standing false alarm is indistinguishable from a real one, so a genuinely
# stuck job would have raised the same flag and been ignored.


def test_stuck_filter_excludes_a_job_that_completed_after_failing(tmp_path):
    from nightly_autorun import Job, stuck_jobs

    runs = tmp_path / "runs"
    marker = runs / "shape-x" / "out.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    job = Job(
        name="shape-x",
        command="shape",
        config="c.yaml",
        run_name="shape-x",
        total=1,
        seeds=[11],
        min_free_gb=1,
        progress_kind="marker",
        requires=None,
        marker="shape-x/out.json",
    )
    state = {"shape-x": {"fails": 99}}
    assert (
        stuck_jobs(runs, [job], state) == set()
    ), "a completed job must never be reported stuck, however many launches failed"


def test_stuck_filter_still_reports_a_job_that_failed_and_is_incomplete(tmp_path):
    from nightly_autorun import Job, stuck_jobs

    runs = tmp_path / "runs"
    runs.mkdir()
    job = Job(
        name="shape-y",
        command="shape",
        config="c.yaml",
        run_name="shape-y",
        total=1,
        seeds=[11],
        min_free_gb=1,
        progress_kind="marker",
        requires=None,
        marker="shape-y/out.json",
    )
    assert stuck_jobs(runs, [job], {"shape-y": {"fails": 99}}) == {"shape-y"}


def test_the_report_does_not_hand_roll_its_own_stuck_scan():
    """One source of truth: the raw comprehension is what drifted."""
    src = (ROOT / "scripts" / "nightly_report.py").read_text(encoding="utf-8")
    assert "na.stuck_jobs(" in src
    assert 's.get("fails", 0) >= na.MAX_FAILS' not in src, (
        "report re-implemented the stuck scan; it must call na.stuck_jobs so the "
        "completeness filter cannot drift out of it again"
    )


# --------------------------------------------------------------------------
# Amendment R.7: BLOCKED is a per-cell status, never an exit that strands the
# queue. On 2026-09-24 a dry-run of the 01:45 tick showed restoration-lag1
# (0.45 GiB) refusing via SystemExit, so no cell queued behind it could launch.


def _gb_job(name, est_gb, total=400, blocked=None):
    return M.Job(
        name=name,
        command="grid",
        config="c.yaml",
        run_name=name,
        total=total,
        seeds=[11, 23, 47, 89],
        min_free_gb=24,
        progress_kind="grid",
        requires=None,
        est_output_gb=est_gb,
        blocked=blocked,
    )


def test_a_held_cell_is_reported_blocked_and_the_next_cell_is_a_candidate(tmp_path):
    jobs = [_gb_job("held", 0.45, blocked="R.5 identity repair"), _gb_job("b", 0.03)]
    candidates, notes = M.launch_candidates(
        tmp_path, jobs, skip=set(), live={}, state={}
    )
    assert [j.name for j, _ in candidates] == ["b"]
    assert "BLOCKED held: R.5 identity repair -> skip" in notes


def test_a_completed_held_cell_is_silent(tmp_path):
    jobs = [_gb_job("held", 0.45, total=1, blocked="x")]
    _write(tmp_path / "held" / "math500" / "m" / "a-256-s11.jsonl", 1)
    candidates, notes = M.launch_candidates(
        tmp_path, jobs, skip=set(), live={}, state={}
    )
    assert candidates == [] and notes == []


def test_load_queue_reads_blocked(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text(
        "version: 1\nenabled: true\njobs:\n"
        "  - {name: a, command: grid, config: c.yaml, run_name: a, total: 1,\n"
        "     seeds: [11], min_free_gb: 1, progress_kind: grid, requires: null,\n"
        "     blocked: 'held for R.5'}\n"
    )
    _, jobs = M.load_queue(q)
    assert jobs[0].blocked == "held for R.5"


def test_deficit_is_exact_and_zero_when_the_cell_fits():
    # headroom 70 - 68.86 = 1.14; need 0.45 + 1.0 margin = 1.45 -> short 0.31
    assert M.disk_deficit_gb(_gb_job("r", 0.45), 68.86, 70.0) == pytest.approx(0.31)
    assert M.disk_deficit_gb(_gb_job("d", 0.03), 68.86, 70.0) == 0.0
    assert M.disk_deficit_gb(_gb_job("n", 0.0), 69.99, 70.0) == 0.0  # no estimate


def test_the_2026_09_24_queue_blocks_the_big_cell_and_admits_the_small_one():
    ok, why = M.disk_admission(_gb_job("restoration-lag1", 0.45), 68.86, 70.0)
    assert not ok
    assert why.startswith("BLOCKED restoration-lag1: storage deficit 0.310 GiB")
    assert f"({int(0.31 * 1024**3)} bytes)" in why
    assert "Nothing is deleted" in why
    ok, _ = M.disk_admission(_gb_job("dacc-t0", 0.03), 68.86, 70.0)
    assert ok


def test_output_still_owed_by_running_cells_is_reserved(tmp_path):
    running, cand = _gb_job("run", 0.10, total=400), _gb_job("cand", 0.10)
    _write(tmp_path / "run" / "math500" / "m" / "a-256-s11.jsonl", 100)  # 1/4 done
    live = {"run": {0}}
    reserved = M.reserved_output_gb(tmp_path, [running, cand], live)
    assert reserved == pytest.approx(0.075)
    # 1.14 headroom: 0.10 + 1.0 = 1.10 fits alone; + 0.075 reserved = 1.175 does not
    assert M.disk_admission(cand, 68.86, 70.0)[0]
    assert not M.disk_admission(cand, 68.86, 70.0, reserved_gb=reserved)[0]


def test_a_tail_filled_cell_is_not_charged_twice(tmp_path):
    running = _gb_job("run", 0.10, total=400)
    _write(tmp_path / "run" / "math500" / "m" / "a-256-s11.jsonl", 100)
    assert M.reserved_output_gb(tmp_path, [running], {"run": {0}}, exclude="run") == 0.0


# --------------------------------------------------------------------------
# R.14: postprocessing stages run inline, idempotently, with retry.


def _gen(name, total=2):
    return M.Job(
        name=name,
        command="grid",
        config="c.yaml",
        run_name=name,
        total=total,
        seeds=[11],
        min_free_gb=24,
        progress_kind="grid",
        requires=None,
    )


def _post(name, *, requires=None, requires_all=()):
    return M.Job(
        name=name,
        command="postprocess",
        config=name,
        run_name=name,
        total=1,
        seeds=[11],
        min_free_gb=0,
        progress_kind="postprocess",
        requires=requires,
        marker=f"postprocess/{name}.json",
        cpu_only=True,
        requires_all=requires_all,
    )


class _Res:
    def __init__(self, rc):
        self.returncode, self.stdout = rc, "line\n"


def _runner(tmp_path, *, terminal=True, rc=0, calls=None):
    def run(argv, **kw):
        if calls is not None:
            calls.append(argv[-1])
        m = tmp_path / "postprocess" / f"{argv[-1]}.json"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({"terminal": terminal, "passed": terminal}))
        return _Res(rc)

    return run


def test_cpu_stage_waits_for_every_dependency(tmp_path):
    jobs = [_gen("a"), _gen("b"), _post("p", requires_all=("a", "b"))]
    _write(tmp_path / "a" / "math500" / "m" / "x-256-s11.jsonl", 2)  # a done
    out = M.run_cpu_stages(tmp_path, jobs, lambda m: None, runner=_runner(tmp_path))
    assert out == [("p", "waiting on ['b']")]
    assert not (tmp_path / "postprocess" / "p.json").exists()


def test_generation_complete_before_start_runs_on_the_first_tick(tmp_path):
    jobs = [_gen("a"), _post("p", requires="a")]
    _write(tmp_path / "a" / "math500" / "m" / "x-256-s11.jsonl", 2)
    calls = []
    out = M.run_cpu_stages(
        tmp_path, jobs, lambda m: None, runner=_runner(tmp_path, calls=calls)
    )
    assert calls == ["p"] and "DONE" in out[0][1]
    # done stages are skipped on later ticks
    M.run_cpu_stages(
        tmp_path, jobs, lambda m: None, runner=_runner(tmp_path, calls=calls)
    )
    assert calls == ["p"]


def test_a_failed_or_partial_stage_is_retried_next_tick(tmp_path):
    jobs = [_gen("a"), _post("p", requires="a")]
    _write(tmp_path / "a" / "math500" / "m" / "x-256-s11.jsonl", 2)
    out = M.run_cpu_stages(
        tmp_path, jobs, lambda m: None, runner=_runner(tmp_path, terminal=False, rc=3)
    )
    assert "retry next tick" in out[0][1] and "rc=3" in out[0][1]
    out = M.run_cpu_stages(tmp_path, jobs, lambda m: None, runner=_runner(tmp_path))
    assert "DONE" in out[0][1]


def test_cpu_stages_never_take_a_gpu(tmp_path):
    jobs = [_post("p")]
    candidates, _ = M.launch_candidates(tmp_path, jobs, skip=set(), live={}, state={})
    assert candidates == []
    job, _ = M.select_job(tmp_path, jobs)
    assert job is None

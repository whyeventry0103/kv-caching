#!/usr/bin/env python3
"""GPU-aware nightly auto-launch WATCHER (remote cron; window 01:45-10:45 Dubai).

Each tick (server = Asia/Dubai, UTC+4): walk the queue in order and launch the FIRST
job that has relaunchable ("needy") seeds -- incomplete AND not owned by a live tmux
session -- onto GPUs that are eligible (>= min_free_gb) AND not occupied by ANY of our
sessions. A job whose remaining seeds are all in flight is skipped in favor of the next
job, so one long tail shard never idles the other GPU. check_remote.sh runs as
preflight; launch goes through scripts/launch.sh in tmux with KVCACHE_MAX_HOURS =
hours-until-10:45. At most one job launches per tick (its seeds may shard across both
GPUs); the 10-min cadence fills the window. Every non-error exits 0 so cron mail flags
only real breakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

NVIDIA_SMI = "/usr/bin/nvidia-smi"
TMUX = "/usr/bin/tmux"
STOP_HOUR = 10  # window stop 10:45 Dubai (start 01:45 is enforced by the cron schedule)
STOP_MINUTE = 45
QUEUE_DEFAULT = PROJECT_ROOT / "configs" / "nightly_queue.yaml"
STATE_PATH = PROJECT_ROOT / "logs" / "nightly_state.json"
MAX_FAILS = 3  # consecutive no-progress launches before a job is marked stuck + skipped
REMOTE_STORAGE = Path("/home/sk10741/kvcache-storage")

BASE_ENV = {
    "KVCACHE_REMOTE": "1",
    "KVCACHE_PERSONAL_FREE_GB": "70",
    "REMOTE_DATA_ROOT": str(REMOTE_STORAGE / "data"),
    "REMOTE_CACHE_ROOT": str(REMOTE_STORAGE / "cache"),
    "REMOTE_RUNS_ROOT": str(REMOTE_STORAGE / "runs"),
    "ENVIRONMENT_NAME": ".venv",
}


@dataclass(frozen=True)
class Job:
    name: str
    command: str
    config: str
    run_name: str
    total: int
    seeds: list[int]
    min_free_gb: int
    progress_kind: str
    requires: str | None
    bench: str = "math500"  # subtree counted for kind 'grid' (e.g. aime24 banks)
    marker: str | None = None  # kind 'marker': done when <runs_root>/<marker> exists
    extra_args: tuple[str, ...] = ()  # appended to launch.sh argv (e.g. download flags)


@dataclass(frozen=True)
class GpuState:
    index: int
    mem_total_mb: int
    mem_free_mb: int
    util_pct: int

    @property
    def free_gb(self) -> float:
        return self.mem_free_mb / 1024


@dataclass(frozen=True)
class LaunchSpec:
    session: str
    gpu_index: int
    seeds: list[int]
    env: dict[str, str]
    runner_path: Path
    launch_argv: list[str]


# ---- pure helpers (table-tested) -------------------------------------------------


def load_queue(path: Path) -> tuple[bool, list[Job]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    enabled = bool(data.get("enabled", True))
    jobs = [
        Job(
            name=j["name"],
            command=j["command"],
            config=j["config"],
            run_name=j["run_name"],
            total=int(j["total"]),
            seeds=[int(s) for s in j["seeds"]],
            min_free_gb=int(j["min_free_gb"]),
            progress_kind=j["progress_kind"],
            requires=j.get("requires"),
            bench=j.get("bench", "math500"),
            marker=j.get("marker"),
            extra_args=tuple(j.get("extra_args") or ()),
        )
        for j in data["jobs"]
    ]
    return enabled, jobs


def parse_nvidia_smi(text: str) -> list[GpuState]:
    gpus = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = [int(p.strip()) for p in line.split(",")]
        gpus.append(GpuState(parts[0], parts[1], parts[2], parts[3]))
    return gpus


def parse_fake_gpus(spec: str) -> list[GpuState]:
    """'0:47000,1:12000' -> GpuStates with that free MB (for offline --dry-run)."""
    gpus = []
    for part in spec.split(","):
        idx, free = part.split(":")
        gpus.append(GpuState(int(idx), 49140, int(free), 0))
    return gpus


def eligible_gpus(gpus: list[GpuState], min_free_gb: int) -> list[GpuState]:
    elig = [g for g in gpus if g.free_gb >= min_free_gb]
    return sorted(elig, key=lambda g: g.mem_free_mb, reverse=True)


def shard_seeds(seeds: list[int], gpus: list[GpuState]) -> dict[int, list[int]]:
    """Round-robin split, freest GPU first (1 GPU -> all seeds; 2 -> 0::2 / 1::2)."""
    n = len(gpus)
    assignment = {}
    for i, gpu in enumerate(gpus):
        shard = seeds[i::n]
        if shard:
            assignment[gpu.index] = shard
    return assignment


def compute_max_hours(
    now: datetime, stop_hour: int = STOP_HOUR, stop_minute: int = STOP_MINUTE
) -> float | None:
    stop = now.replace(hour=stop_hour, minute=stop_minute, second=0, microsecond=0)
    hours = (stop - now).total_seconds() / 3600
    if hours < 1:
        return None  # past stop or <1h left -> skip (don't run a tiny window)
    # 2-decimal floor: uses the (near-)full window without ever overshooting the stop
    # (an int floor lost up to ~1h when cron fired a second after the start).
    return math.floor(hours * 100) / 100


def count_progress(runs_root: Path, job: Job) -> int:
    run_dir = runs_root / job.run_name
    if job.progress_kind == "marker":
        # Gate jobs (e.g. the rkv generation smoke): done exactly when the PASS
        # marker exists under the base runs root; there are no row files to count.
        return 1 if job.marker and (runs_root / job.marker).exists() else 0
    if job.progress_kind == "grid":
        base = run_dir / job.bench
        paths = (
            [p for p in base.rglob("*.jsonl") if "token_ids" not in p.parts]
            if base.exists()
            else []
        )
    elif job.progress_kind == "readout":
        base = run_dir / "readout"
        paths = list(base.rglob("*.jsonl")) if base.exists() else []
    else:
        raise ValueError(f"unknown progress_kind: {job.progress_kind}")
    return sum(
        1
        for p in paths
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def count_per_seed(runs_root: Path, job: Job) -> dict[int, int]:
    """Completed records per seed (from `...-s<seed>.jsonl`), summed over presses."""
    if job.progress_kind == "marker":
        done = count_progress(runs_root, job)
        return {job.seeds[0]: done} if done else {}
    sub = job.bench if job.progress_kind == "grid" else "readout"
    base = runs_root / job.run_name / sub
    counts: dict[int, int] = {}
    if not base.exists():
        return counts
    for p in base.rglob("*.jsonl"):
        if "token_ids" in p.parts:
            continue
        m = re.search(r"-s(\d+)\.jsonl$", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        lines = p.read_text(encoding="utf-8").splitlines()
        counts[seed] = counts.get(seed, 0) + sum(1 for ln in lines if ln.strip())
    return counts


def session_gpus(run_name: str) -> set[int]:
    """GPU indices with a LIVE tmux session for this run (`<run>-g<idx>`)."""
    result = subprocess.run([TMUX, "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return set()
    gpus = set()
    for line in result.stdout.splitlines():
        m = re.match(rf"^{re.escape(run_name)}-g(\d+):", line)
        if m:
            gpus.add(int(m.group(1)))
    return gpus


def live_map(run_names: list[str], session_names: list[str]) -> dict[str, set[int]]:
    """run_name -> live GPU indices, from one parse of tmux session names."""
    live: dict[str, set[int]] = {rn: set() for rn in run_names}
    for name in session_names:
        for rn in run_names:
            m = re.match(rf"^{re.escape(rn)}-g(\d+)$", name)
            if m:
                live[rn].add(int(m.group(1)))
    return live


def tmux_session_names() -> list[str]:
    result = subprocess.run([TMUX, "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in result.stdout.splitlines() if ":" in line]


def launch_candidates(
    runs_root: Path,
    jobs: list[Job],
    *,
    skip: set[str],
    live: dict[str, set[int]],
    state: dict[str, dict] | None = None,
) -> tuple[list[tuple[Job, list[int]]], list[str]]:
    """Jobs with relaunchable (needy) seeds, in queue order, plus skip notes.

    A job whose incomplete seeds are ALL in flight is skipped (not returned) so a
    later job can fill the otherwise-idle GPU during the first job's tail shard —
    previously the tick no-op'd there and one GPU could idle for hours.
    """
    notes: list[str] = []
    by_name = {j.name: j for j in jobs}
    state = load_state() if state is None else state
    candidates: list[tuple[Job, list[int]]] = []
    for job in jobs:
        if job.name in skip:
            continue
        if count_progress(runs_root, job) >= job.total:
            continue
        if job.requires is not None:
            req = by_name.get(job.requires)
            if req is None or count_progress(runs_root, req) < req.total:
                notes.append(f"{job.name}: dependency incomplete -> skip")
                continue
        per_seed = count_per_seed(runs_root, job)
        seed_total = max(1, job.total // len(job.seeds))
        incomplete = [s for s in job.seeds if per_seed.get(s, 0) < seed_total]
        shards = state.get(job.name, {}).get("shards", {})
        job_live = live.get(job.run_name, set())
        running = {s for g in job_live for s in shards.get(str(g), [])}
        needy = [s for s in incomplete if s not in running]
        if not needy:
            notes.append(
                f"{job.name}: incomplete seeds all in flight on "
                f"{sorted(job_live)} -> try next job"
            )
            continue
        candidates.append((job, needy))
    return candidates, notes


def select_job(
    runs_root: Path, jobs: list[Job], skip: set[str] | None = None
) -> tuple[Job | None, str]:
    skip = skip or set()
    by_name = {j.name: j for j in jobs}
    for job in jobs:
        if job.name in skip:
            continue  # marked stuck (repeated no-progress failures) -> don't reselect
        if count_progress(runs_root, job) >= job.total:
            continue
        if job.requires is not None:
            req = by_name.get(job.requires)
            if req is None:
                return None, f"{job.name} requires unknown job {job.requires}"
            if count_progress(runs_root, req) < req.total:
                continue  # dependency not complete; skip tonight
        return job, "selected"
    if all(count_progress(runs_root, j) >= j.total for j in jobs if j.name not in skip):
        return None, "all jobs complete"
    return None, "next job blocked on incomplete dependency or stuck"


def build_launch_specs(
    job: Job, assignment: dict[int, list[int]], max_hours: float
) -> list[LaunchSpec]:
    specs = []
    for gpu_index, seeds in sorted(assignment.items()):
        session = f"{job.run_name}-g{gpu_index}"
        env = dict(BASE_ENV)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["KVCACHE_RUN_NAME"] = job.run_name
        env["KVCACHE_MAX_HOURS"] = str(max_hours)
        env["KVCACHE_ALLOW_EXISTING_RUN"] = "1"
        launch_argv = [
            "bash",
            "scripts/launch.sh",
            "--i-understand",
            job.command,
            job.config,
            "--resume",
            "--seeds",
            ",".join(str(s) for s in seeds),
            *job.extra_args,
        ]
        specs.append(
            LaunchSpec(
                session=session,
                gpu_index=gpu_index,
                seeds=seeds,
                env=env,
                runner_path=PROJECT_ROOT / "logs" / f"{session}.runner.sh",
                launch_argv=launch_argv,
            )
        )
    return specs


# ---- side-effecting helpers ------------------------------------------------------


def query_gpus() -> list[GpuState]:
    result = subprocess.run(
        [
            NVIDIA_SMI,
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed ({result.returncode}): {result.stderr}")
    return parse_nvidia_smi(result.stdout)


def run_preflight(job: Job) -> tuple[bool, str]:
    env = {**os.environ, **BASE_ENV, "KVCACHE_ALLOW_EXISTING_RUN": "1"}
    argv = [
        "bash",
        "scripts/check_remote.sh",
        "--config",
        job.config,
        "--run-name",
        job.run_name,
    ]
    result = subprocess.run(
        argv, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def write_runner(spec: LaunchSpec) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key, value in spec.env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f"cd {shlex.quote(str(PROJECT_ROOT))}")
    lines.append("exec " + " ".join(shlex.quote(a) for a in spec.launch_argv))
    spec.runner_path.parent.mkdir(parents=True, exist_ok=True)
    spec.runner_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec.runner_path.chmod(0o700)


def launch_session(spec: LaunchSpec) -> None:
    runner = shlex.quote(str(spec.runner_path))
    subprocess.run(
        [TMUX, "new-session", "-d", "-s", spec.session, f"bash {runner}"],
        check=True,
    )


def load_state() -> dict[str, dict]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, dict]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def our_sessions_running(run_names: list[str]) -> list[str]:
    """tmux sessions launched by us (named '<run_name>-g<idx>'); empty => GPUs free.

    The watcher launches only when this is empty, so it never collides with a job in
    progress and never double-launches -- it just fills the window when work finishes.
    """
    result = subprocess.run([TMUX, "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return []  # no tmux server -> nothing of ours is running
    running = []
    for line in result.stdout.splitlines():
        name = line.split(":", 1)[0]
        if any(name.startswith(f"{rn}-g") for rn in run_names):
            running.append(name)
    return running


# ---- main ------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore the night marker")
    parser.add_argument("--stop-hour", type=int, default=STOP_HOUR)
    parser.add_argument("--stop-minute", type=int, default=STOP_MINUTE)
    parser.add_argument("--now", default="", help="ISO datetime override (testing)")
    parser.add_argument("--fake-gpus", default="", help="'0:47000,1:12000' (test)")
    parser.add_argument("--skip-preflight", action="store_true", help="dry-run only")
    parser.add_argument(
        "--max-hours-override",
        type=float,
        default=0.0,
        help="override computed max-hours (supervised live test, e.g. 0.2)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    now = datetime.fromisoformat(args.now) if args.now else datetime.now()

    def log(msg: str) -> None:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

    log(f"=== nightly_autorun start (dry_run={args.dry_run} force={args.force}) ===")

    enabled, jobs = load_queue(args.queue)
    if not enabled:
        log("queue enabled: false (kill-switch) -> exit")
        return 0

    runs_root = Path(BASE_ENV["REMOTE_RUNS_ROOT"])
    state = load_state()
    stuck = {n for n, s in state.items() if s.get("fails", 0) >= MAX_FAILS}
    if stuck:
        log(f"stuck jobs (skipped, {MAX_FAILS}+ no-progress launches): {sorted(stuck)}")

    # One tmux parse -> live GPUs per run + globally occupied GPUs. A job whose
    # remaining seeds are all in flight no longer blocks the queue: later jobs may
    # fill the otherwise-idle GPU during its tail shard (launch_candidates).
    session_names = [] if (args.force or args.dry_run) else tmux_session_names()
    live = live_map([j.run_name for j in jobs], session_names)
    occupied = {g for gpu_set in live.values() for g in gpu_set}
    candidates, notes = launch_candidates(
        runs_root, jobs, skip=stuck, live=live, state=state
    )
    for note in notes:
        log(note)
    if not candidates:
        log("no launchable job (complete, stuck, dep-gated, or in flight) -> exit")
        return 0

    max_hours = args.max_hours_override or compute_max_hours(
        now, args.stop_hour, args.stop_minute
    )
    if not max_hours:
        log(f"<1h to {args.stop_hour}:{args.stop_minute:02d} stop -> no launch")
        return 0

    gpus = parse_fake_gpus(args.fake_gpus) if args.fake_gpus else query_gpus()
    for g in gpus:
        log(f"  gpu{g.index}: {g.free_gb:.1f}GB free util={g.util_pct}%")

    for job, needy in candidates:
        free = [
            g for g in eligible_gpus(gpus, job.min_free_gb) if g.index not in occupied
        ]
        if not free:
            log(
                f"{job.name}: no free GPU >= {job.min_free_gb}GB "
                f"(occupied={sorted(occupied)}) -> try next job"
            )
            continue
        done = count_progress(runs_root, job)
        log(
            f"selected: {job.name} {done}/{job.total} "
            f"(min_free_gb={job.min_free_gb}); needy seeds {needy}; "
            f"window->{args.stop_hour}:{args.stop_minute:02d}"
        )
        assignment = shard_seeds(needy, free)  # split needy seeds across free GPUs
        log(f"launch assignment: {assignment}")
        specs = build_launch_specs(job, assignment, max_hours)

        if not (args.dry_run and args.skip_preflight):
            ok, output = run_preflight(job)
            log(f"preflight check_remote.sh: {'PASS' if ok else 'FAIL'}")
            if not ok:
                log("preflight output:\n" + output)
                if not args.dry_run:
                    log("ABORT: preflight failed -> no launch")
                    return 0

        if args.dry_run:
            log("DRY-RUN would launch:")
            for spec in specs:
                log(
                    f"  tmux -s {spec.session}: {' '.join(spec.launch_argv)} "
                    f"[CUDA={spec.gpu_index}, MAX_HOURS={max_hours}]"
                )
            log("DRY-RUN: nothing launched")
            return 0

        # Crash-loop guard (whole-job, committed launch only): if TOTAL progress did
        # not advance since the last launch, nothing is progressing -> count a
        # failure; at MAX_FAILS mark stuck, then let the NEXT candidate use the GPUs
        # this same tick. (A live shard making progress resets the counter, so
        # recovery isn't mistaken for a crash.)
        per_seed = count_per_seed(runs_root, job)
        progressed = sum(per_seed.values())
        prev = state.get(job.name, {})
        fails = (
            prev.get("fails", 0) + 1
            if progressed <= prev.get("last_progress", -1)
            else 0
        )
        if fails >= MAX_FAILS:
            state[job.name] = {**prev, "last_progress": progressed, "fails": fails}
            save_state(state)
            log(
                f"STUCK: {job.name} no progress across {fails} launches (at "
                f"{progressed}/{job.total}) -> skipping; investigate "
                f"logs/{job.run_name}.log"
            )
            continue

        for spec in specs:
            write_runner(spec)
            launch_session(spec)
            log(f"launched {spec.session} (gpu {spec.gpu_index}, seeds {spec.seeds})")
        shards = state.get(job.name, {}).get("shards", {})
        job_live = live.get(job.run_name, set())
        new_shards = {str(g): shards[str(g)] for g in job_live if str(g) in shards}
        for gpu, seeds in assignment.items():
            new_shards[str(gpu)] = seeds
        state[job.name] = {
            "last_progress": progressed,
            "fails": fails,
            "shards": new_shards,
        }
        save_state(state)
        log("=== done ===")
        return 0

    log("no candidate fits the free GPUs -> wait")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

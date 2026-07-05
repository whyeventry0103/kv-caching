#!/usr/bin/env python3
"""Daily status email after the nightly session (runs on the remote via cron ~10:55).

Reports: the whole queue with per-job progress, what ran last night, total progress % to
queue completion (and whether the paper-core jobs are done), and any crash/stuck alert.
Sends via direct-to-MX delivery to an @lums.edu.pk address (the remote is on the LUMS
network; the O365 MX accepts inbound mail for the domain, no credentials needed).

--dry-run prints the report and does NOT send. It is wrapped so a failure still emails
an error (a silent cron failure would mean no status at all).
"""

from __future__ import annotations

import argparse
import importlib.util
import smtplib
import socket
import sys
import traceback
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reuse the orchestrator's queue/progress logic (single source of truth, no drift).
_spec = importlib.util.spec_from_file_location(
    "nightly_autorun", PROJECT_ROOT / "scripts" / "nightly_autorun.py"
)
na = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("nightly_autorun", na)
_spec.loader.exec_module(na)

TO_ADDR = "27100172@lums.edu.pk"
MX = "lums-edu-pk.mail.protection.outlook.com"
# Jobs that constitute the writable paper core (256 decomposition + bracket + Figure 4).
CORE_JOBS = {"grid-diss256-tok", "reverse-readout-diss256", "replay-diss256"}
PREV_SNAPSHOT = PROJECT_ROOT / "logs" / "nightly_report_prev.json"
# Scope decision 2026-07-03 (context.md): R-KV cells + AIME are cut to future work if
# not producing rows by this date; the countdown keeps the decision visible.
DROP_DEAD = datetime(2026, 8, 1)
DROP_DEAD_LABEL = "R-KV cells + AIME"
BAR_WIDTH = 20


def _bar(done: int, total: int) -> str:
    filled = int(BAR_WIDTH * min(1.0, done / total)) if total else 0
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def generate_report(write_snapshot: bool = True) -> tuple[str, str]:
    """Return (subject, body). Never raises -- failures are captured into the body.

    ``write_snapshot=False`` (dry runs) leaves the delta baseline untouched so a
    manual preview never shrinks the next morning's "+N last night" numbers.
    """
    import json

    runs_root = Path(na.BASE_ENV["REMOTE_RUNS_ROOT"])
    enabled, jobs = na.load_queue(na.QUEUE_DEFAULT)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    progress = {j.name: na.count_progress(runs_root, j) for j in jobs}
    total_done = sum(progress.values())
    total_need = sum(j.total for j in jobs)
    pct = 100 * total_done / total_need if total_need else 0.0

    core = [j for j in jobs if j.name in CORE_JOBS]
    core_done = all(progress[j.name] >= j.total for j in core)
    core_dn = sum(progress[j.name] for j in core)
    core_tot = sum(j.total for j in core)

    # per-job deltas vs yesterday's snapshot (old snapshots lack "progress": tolerate)
    prev = json.loads(PREV_SNAPSHOT.read_text()) if PREV_SNAPSHOT.exists() else {}
    prev_progress = prev.get("progress", {})
    delta_by_job = {
        n: progress[n] - prev_progress.get(n, progress[n]) for n in progress
    }
    delta = (
        total_done - prev.get("total_done", total_done)
        if prev
        else sum(delta_by_job.values())
    )
    finished_last_night = [
        j.name
        for j in jobs
        if progress[j.name] >= j.total > prev_progress.get(j.name, j.total)
    ]

    # crash/stuck detection + what launched, from state and today's watcher log
    state = na.load_state()
    stuck = [n for n, s in state.items() if s.get("fails", 0) >= na.MAX_FAILS]
    autorun_log = PROJECT_ROOT / "logs" / "nightly_autorun.log"
    stuck_today: list[str] = []
    launched_today: list[str] = []
    if autorun_log.exists():
        for line in autorun_log.read_text(encoding="utf-8").splitlines():
            if today in line and "STUCK:" in line:
                stuck_today.append(line.split("STUCK:")[1].strip())
            if today in line and "launched " in line:
                launched_today.append(line.split("launched ")[1].split(" (")[0])

    alert = bool(stuck or stuck_today)
    flag = "ATTENTION" if alert else ("CORE DONE" if core_done else "OK")
    subject = f"[kvcache] {today} | +{delta} last night | {pct:.0f}% queue | {flag}"

    lines = [
        f"KVCACHE NIGHTLY -- {today}  (queue {'ENABLED' if enabled else 'OFF'}, "
        f"flag {flag})",
        "=" * 66,
        "",
        "LAST NIGHT",
        f"  sessions launched : {', '.join(sorted(set(launched_today))) or 'none'}",
        f"  rows produced     : +{delta}",
    ]
    for name, d in sorted(delta_by_job.items(), key=lambda kv: -kv[1]):
        if d > 0:
            star = "  ** COMPLETED **" if name in finished_last_night else ""
            lines.append(f"    {name:<26} +{d}{star}")
    if not any(d > 0 for d in delta_by_job.values()):
        lines.append("    (no new rows -- check window/GPU/stuck below)")

    lines += ["", "QUEUE", f"  {'job':<26} {'done/total':>11}  progress"]
    for j in jobs:
        d = progress[j.name]
        mark = "DONE " if d >= j.total else ("STUCK" if j.name in stuck else "     ")
        lines.append(
            f"  {j.name:<26} {f'{d}/{j.total}':>11}  {_bar(d, j.total)} {mark}"
        )
    remaining = total_need - total_done
    eta = ""
    if remaining and delta > 0:
        eta = f"  (~{-(-remaining // delta)} nights at last night's rate)"
    lines += [
        f"  TOTAL {total_done}/{total_need} = {pct:.1f}%"
        f"  (+{delta} last night){eta}",
        f"  PAPER CORE (256 decomp+bracket+Fig4): {core_dn}/{core_tot} -> "
        + ("COMPLETE" if core_done else "in progress"),
    ]

    # Tonight's plan: what the watcher will pick at 01:45 (no sessions run at report
    # time, so live is read fresh; per-seed detail doubles as a looping-shard flag).
    try:
        live = na.live_map([j.run_name for j in jobs], na.tmux_session_names())
        candidates, _ = na.launch_candidates(
            runs_root, jobs, skip=set(stuck), live=live, state=state
        )
        lines += ["", "TONIGHT (01:45-10:45 Dubai)"]
        if candidates:
            for job, needy in candidates[:2]:
                lines.append(
                    f"  next: {job.name} (min_free {job.min_free_gb}GB), "
                    f"needy seeds {needy}"
                )
            per_seed = na.count_per_seed(runs_root, candidates[0][0])
            if per_seed:
                seed_total = max(
                    1, candidates[0][0].total // len(candidates[0][0].seeds)
                )
                detail = " ".join(
                    f"s{s}={per_seed.get(s, 0)}/{seed_total}"
                    for s in candidates[0][0].seeds
                )
                lines.append(f"  per-seed ({candidates[0][0].name}): {detail}")
        else:
            lines.append("  nothing launchable (complete, stuck, or dep-gated)")
    except Exception as exc:  # noqa: BLE001 -- plan is best-effort, never fatal
        lines += ["", f"TONIGHT: (plan unavailable: {exc})"]

    days_left = (DROP_DEAD - now).days
    lines += [
        "",
        "DEADLINES",
        f"  {DROP_DEAD_LABEL} drop-dead {DROP_DEAD:%Y-%m-%d}: D-{max(days_left, 0)}"
        + ("  ** PASSED -- cut to future work **" if days_left < 0 else ""),
    ]

    lines += ["", "HEALTH"]
    if alert:
        lines.append("  *** CRASH/STUCK ALERT ***")
        for s in stuck_today or [f"{n} (state)" for n in stuck]:
            lines.append(f"    - {s}")
        lines.append(
            "    -> hit MAX_FAILS no-progress launches; investigate "
            "logs/<run>.log on the server"
        )
    else:
        lines.append("  no crashes / stuck jobs")
    try:
        gpus = na.query_gpus()
        lines.append(
            "  gpus: "
            + " | ".join(f"gpu{g.index} {g.free_gb:.0f}GB free" for g in gpus)
        )
    except Exception:
        pass
    try:
        import shutil

        usage = shutil.disk_usage(na.REMOTE_STORAGE)
        lines.append(f"  disk: {usage.free / 1e9:.0f}GB free filesystem-wide")
    except Exception:
        pass

    if write_snapshot:
        PREV_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        PREV_SNAPSHOT.write_text(
            json.dumps({"total_done": total_done, "date": today, "progress": progress})
        )
    return subject, "\n".join(lines)


def send_email(subject: str, body: str) -> str:
    """Authenticated SMTP if ~/.kvcache_smtp.json exists (reliable, e.g. Gmail app-pw);
    else fall back to direct-to-MX. Returns the recipient actually used."""
    import json
    from email.utils import formatdate, make_msgid
    from pathlib import Path

    creds_path = Path.home() / ".kvcache_smtp.json"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)

    if creds_path.exists():  # {host,port,user,password,to}
        c = json.loads(creds_path.read_text())
        to = c.get("to", TO_ADDR)
        msg["From"] = c["user"]
        msg["To"] = to
        msg["Message-ID"] = make_msgid(domain=c["user"].split("@")[-1])
        with smtplib.SMTP(c["host"], int(c["port"]), timeout=40) as s:
            s.ehlo()
            s.starttls()
            s.login(c["user"], c["password"])
            s.send_message(msg)
        return to
    host = socket.gethostname()
    msg["From"] = f"kvcache-autorun@{host}.lums.edu.pk"
    msg["To"] = TO_ADDR
    msg["Message-ID"] = make_msgid(domain=f"{host}.lums.edu.pk")
    with smtplib.SMTP(MX, 25, timeout=40) as s:
        s.ehlo()
        try:
            s.starttls()
            s.ehlo()
        except smtplib.SMTPException:
            pass
        s.send_message(msg)
    return TO_ADDR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = parser.parse_args()
    try:
        subject, body = generate_report(write_snapshot=not args.dry_run)
    except Exception:
        subject = "[kvcache] STATUS REPORT FAILED"
        body = "nightly_report.py crashed:\n\n" + traceback.format_exc()
    print(subject)
    print(body)
    if not args.dry_run:
        recipient = send_email(subject, body)
        print(f"\n[sent to {recipient}]")
    else:
        print("\n[DRY RUN -- not sent]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

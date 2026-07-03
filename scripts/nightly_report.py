#!/usr/bin/env python3
"""Daily status email after the nightly session (runs on the remote via cron ~09:05).

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


def generate_report() -> tuple[str, str]:
    """Return (subject, body). Never raises -- failures are captured into the body."""
    import json

    runs_root = Path(na.BASE_ENV["REMOTE_RUNS_ROOT"])
    enabled, jobs = na.load_queue(na.QUEUE_DEFAULT)
    today = datetime.now().strftime("%Y-%m-%d")

    progress = {j.name: na.count_progress(runs_root, j) for j in jobs}
    total_done = sum(progress.values())
    total_need = sum(j.total for j in jobs)
    pct = 100 * total_done / total_need if total_need else 0.0

    core = [j for j in jobs if j.name in CORE_JOBS]
    core_done = all(progress[j.name] >= j.total for j in core)
    core_dn = sum(progress[j.name] for j in core)
    core_tot = sum(j.total for j in core)

    # delta since last report
    prev = json.loads(PREV_SNAPSHOT.read_text()) if PREV_SNAPSHOT.exists() else {}
    delta = total_done - prev.get("total_done", total_done)

    # crash/stuck detection
    state = na.load_state()
    stuck = [n for n, s in state.items() if s.get("fails", 0) >= na.MAX_FAILS]
    autorun_log = PROJECT_ROOT / "logs" / "nightly_autorun.log"
    stuck_today = []
    launched_today = []
    if autorun_log.exists():
        for line in autorun_log.read_text(encoding="utf-8").splitlines():
            if today in line and "STUCK:" in line:
                stuck_today.append(line.split("STUCK:")[1].strip())
            if today in line and "launched " in line:
                launched_today.append(line.split("launched ")[1].split(" (")[0])

    alert = bool(stuck or stuck_today)
    flag = "ATTENTION" if alert else ("CORE DONE" if core_done else "OK")
    subject = f"[kvcache] {today} status: {pct:.0f}% queue, {flag}"

    lines = [
        f"KVCACHE NIGHTLY STATUS ({today})  queue {'ENABLED' if enabled else 'OFF'}",
        "",
        f"TOTAL PROGRESS: {total_done}/{total_need} = {pct:.1f}% of full queue"
        f"  (+{delta} since last report)",
        f"PAPER CORE (256 decomp+bracket+Fig4): {core_dn}/{core_tot} -> "
        + ("COMPLETE - can start writing core results" if core_done else "in progress"),
        "",
        "QUEUE:",
    ]
    for j in jobs:
        d = progress[j.name]
        mark = "DONE" if d >= j.total else ("STUCK" if j.name in stuck else "....")
        lines.append(f"  [{mark:>5}] {j.name:<26} {d}/{j.total}")
    lines += ["", f"LAST NIGHT launched: {sorted(set(launched_today)) or 'nothing'}"]

    if alert:
        lines += ["", "*** CRASH/STUCK ALERT ***"]
        for s in stuck_today or [f"{n} (state)" for n in stuck]:
            lines.append(f"  - {s}")
        lines.append(
            "  -> a job hit MAX_FAILS no-progress launches and was skipped; "
            "investigate logs/<run>.log on the server"
        )
    else:
        lines += ["", "No crashes/stuck jobs."]

    try:
        gpus = na.query_gpus()
        lines += [
            "",
            "GPUs now: "
            + " | ".join(f"gpu{g.index} {g.free_gb:.0f}GB free" for g in gpus),
        ]
    except Exception:
        pass

    PREV_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    PREV_SNAPSHOT.write_text(json.dumps({"total_done": total_done, "date": today}))
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
        subject, body = generate_report()
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

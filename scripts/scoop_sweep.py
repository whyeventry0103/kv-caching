#!/usr/bin/env python3
"""Weekly arXiv scoop sweep (Mondays, remote cron).

Queries the arXiv API for the preregistered foreclosure-watch terms, keeps entries
submitted in the last --days days, and emails a digest via ~/.kvcache_smtp.json
(same transport file as nightly_report). Falls back to stdout when the creds file
is absent (e.g. local dry runs). Log of verdicts lives in notes/scoop-watch.md
(maintained manually after reading the digest; the foreclosure test is there).

Cron (remote): 20 9 * * 1  cd ~/kv-caching && .venv/bin/python scripts/scoop_sweep.py
"""

from __future__ import annotations

import argparse
import json
import smtplib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, UTC
from email.message import EmailMessage
from pathlib import Path

QUERIES = [
    "KV cache reasoning",
    "compression evaluation reasoning",
    "teacher forcing compression",
    "KV cache compression readout trajectory",
    "KV cache free-running evaluation",
]
ATOM = "{http://www.w3.org/2005/Atom}"
API = "http://export.arxiv.org/api/query"


def fetch(query: str, max_results: int = 15) -> list[dict[str, str]]:
    params = urllib.parse.urlencode(
        {
            "search_query": f'all:"{query}"',
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }
    )
    with urllib.request.urlopen(f"{API}?{params}", timeout=60) as response:
        tree = ET.fromstring(response.read())
    entries = []
    for entry in tree.findall(f"{ATOM}entry"):
        entries.append(
            {
                "id": entry.findtext(f"{ATOM}id", ""),
                "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
                "published": entry.findtext(f"{ATOM}published", ""),
                "query": query,
            }
        )
    return entries


def sweep(days: int) -> str:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    seen: dict[str, dict[str, str]] = {}
    errors = []
    for query in QUERIES:
        try:
            for entry in fetch(query):
                published = entry["published"]
                try:
                    when = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if when >= cutoff and entry["id"] not in seen:
                    seen[entry["id"]] = entry
        except Exception as exc:  # noqa: BLE001 -- a dead mirror must not kill cron
            errors.append(f"{query}: {exc}")
    lines = [
        f"arXiv scoop sweep (last {days} days) -- {len(seen)} new entries.",
        "Foreclosure test: does it separate trajectory from readout with an",
        "explicit instrument? Verdicts go in notes/scoop-watch.md.",
        "",
    ]
    for entry in sorted(seen.values(), key=lambda e: e["published"], reverse=True):
        lines.append(f"- {entry['published'][:10]}  {entry['title']}")
        lines.append(f"  {entry['id']}  (query: {entry['query']})")
    if not seen:
        lines.append("(no new matches this week)")
    if errors:
        lines.append("")
        lines.append("Query errors: " + "; ".join(errors))
    return "\n".join(lines)


def send_email(body: str, creds_path: Path) -> bool:
    if not creds_path.exists():
        return False
    creds = json.loads(creds_path.read_text())
    message = EmailMessage()
    message["Subject"] = "kvcache scoop sweep (weekly)"
    message["From"] = creds["user"]
    message["To"] = creds["to"]
    message.set_content(body)
    with smtplib.SMTP(creds["host"], int(creds["port"]), timeout=60) as smtp:
        smtp.starttls()
        smtp.login(creds["user"], creds["password"])
        smtp.send_message(message)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="print, never email")
    args = parser.parse_args()
    body = sweep(args.days)
    if not args.dry_run and send_email(body, Path.home() / ".kvcache_smtp.json"):
        print("scoop sweep emailed.")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

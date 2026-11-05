#!/usr/bin/env python3
"""Record which of our formative submissions never started.

Writes reports/paiec_v5_report/data/unscored_runs.json with the id, archive,
creation time and status of every submission that is still waiting in the
platform queue. Cancelled runs and runs that finished are left out.

usage: queue_snapshot.py
"""

import datetime
import json
from pathlib import Path

import codabench_client as cb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/paiec_v5_report/data/unscored_runs.json"
WAITING = {"Submitted", "Preparing", "Running", "Scoring"}


def main():
    token = cb.load_token()
    rows = cb.api(f"submissions/?phase={cb.PHASE_ID}&page_size=100", token=token)["results"]
    waiting = [{"id": r["id"], "archive": r["filename"], "created": r["created_when"][:16] + "Z",
                "status": r["status"]} for r in rows if r["status"] in WAITING]
    snap = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "waiting": sorted(waiting, key=lambda r: r["id"])}
    OUT.write_text(json.dumps(snap, indent=1) + "\n")
    print(json.dumps(snap))


if __name__ == "__main__":
    main()

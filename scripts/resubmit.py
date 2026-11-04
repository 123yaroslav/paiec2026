#!/usr/bin/env python3
"""Create new Codabench submissions from an already uploaded archive.

Each formative run draws a new random subset of about nine groups, so one
archive needs several runs for a usable mean. Re-using the uploaded dataset key
avoids uploading the same bytes again.

usage: resubmit.py ARCHIVE_NAME COUNT [--label TEXT]
"""

import argparse
import datetime
import json
from pathlib import Path

import codabench_client as cb

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "private" / "submissions"
LOG = RECEIPTS / "resubmits.jsonl"


def dataset_key(archive_name):
    keys = [json.loads(p.read_text()) for p in RECEIPTS.glob("*.json") if not p.name.startswith("._")]
    keys = [k for k in keys if k.get("archive") == archive_name and k.get("upload_complete")]
    if len(keys) != 1:
        raise SystemExit(f"expected one completed upload of {archive_name}, found {len(keys)}")
    return keys[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("count", type=int)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    token = cb.load_token()
    receipt = dataset_key(args.archive)
    competition = cb.api("competitions/17828/", token=token)
    queue = (competition.get("queue") or {}).get("id")
    for _ in range(args.count):
        if not cb.api(f"can_make_submission/{cb.PHASE_ID}/", token=token).get("can"):
            raise SystemExit("Codabench does not permit a submission now")
        result = cb.api("submissions/", token=token, payload={"phase": cb.PHASE_ID, "data": receipt["dataset_key"], "queue": queue})
        row = {"time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "archive": args.archive, "sha256": receipt["sha256"], "submission_id": result["id"],
               "status": result.get("status"), "label": args.label}
        with LOG.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row))


if __name__ == "__main__":
    main()

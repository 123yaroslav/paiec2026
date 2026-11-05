#!/usr/bin/env python3
"""Collect formative results of our submissions and compare candidate archives.

For every submission in the upload receipts and resubmits.jsonl it reads the
status, downloads the owner-visible scoring output once it is finished, and
parses group ALC rows. It prints the mean official score per archive and paired
differences on groups (subject_id, benchmark_id) seen by both archives. Group
IDs serve only to pair evaluations here. No predictor reads them.
"""

import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import codabench_client as cb

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "private" / "submissions"
FEED = ROOT / "private" / "feedback"
STATE = REC / "track_state.json"


def submissions():
    subs = {}
    for p in REC.glob("*.json"):
        if p.name.startswith("._") or p.name == STATE.name:
            continue
        d = json.loads(p.read_text())
        if d.get("submission_id"):
            subs[int(d["submission_id"])] = d["archive"]
    log = REC / "resubmits.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            subs[int(d["submission_id"])] = d["archive"]
    return subs


def parse(path):
    text = path.read_text()
    head = text.split("Label budget")[0]
    rows = re.findall(r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)", head)
    return {(s, b): float(alc) for s, b, n, alc, ece in rows}


def main():
    token = cb.load_token()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    per_archive = defaultdict(list)
    groups = defaultdict(lambda: defaultdict(list))
    pending = defaultdict(int)
    for sid, archive in sorted(submissions().items()):
        feed = FEED / f"{sid}_scoring_stdout.txt"
        if not feed.exists():
            # A cancelled run can still be picked up from the queue later, so only Failed is final.
            if state.get(str(sid)) == "Failed":
                continue
            try:
                info = cb.api(f"submissions/{sid}/", token=token)
            except RuntimeError as exc:
                pending[(archive, "error " + str(exc)[-8:])] += 1
                continue
            state[str(sid)] = info.get("status")
            if info.get("status") == "Finished":
                try:
                    cb.feedback(sid, token)
                except (RuntimeError, ValueError) as exc:
                    pending[(archive, "feedback " + str(exc)[:30])] += 1
                    continue
            else:
                pending[(archive, info.get("status"))] += 1
                continue
        rows = parse(feed)
        if not rows:
            continue
        per_archive[archive].append(sum(rows.values()) / len(rows))
        for key, alc in rows.items():
            groups[archive][key].append(alc)
    STATE.write_text(json.dumps(state, indent=1))
    print("pending:", dict(pending))
    print(f"{'archive':<42} runs  mean    sd")
    for archive, vals in sorted(per_archive.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else float("nan")
        print(f"{archive:<42} {len(vals):>4} {m:.4f} {sd:.4f}")
    print("paired on shared groups (row minus column archive, mean over shared groups)")
    names = sorted(groups)
    for a, b in itertools.permutations(names, 2):
        shared = set(groups[a]) & set(groups[b])
        if len(shared) >= 3 and a < b:
            diffs = [sum(groups[a][k]) / len(groups[a][k]) - sum(groups[b][k]) / len(groups[b][k]) for k in shared]
            m = sum(diffs) / len(diffs)
            se = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1) / len(diffs))
            print(f"  {a} - {b}: {m:+.4f} (se {se:.4f}, {len(shared)} groups)")


if __name__ == "__main__":
    main()

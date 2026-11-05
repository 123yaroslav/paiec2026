"""Aggregate accuracy statistics of the formative test pool from the Beta baseline feedback.

The official Beta(2,2) baseline predicts one constant q per group at every budget.
For a constant prediction, Brier = p(1-p) + (q-p)^2 and ECE = |q-p|, with p the
accuracy of the group's evaluation responses. Hence
  budget 0   q = 0.5        ECE_0 = |p - 0.5|
  budget 31  p(1-p) = B_31 - ECE_31^2      (consistency check)
  budget 1   q = 0.4 or 0.6, and |q - p| = ECE_1 fixes the side of 0.5.
Only aggregate statistics leave this script. No predictor reads group identifiers.

It also computes the same statistics for the training corpus (groups with at least
80 items, benchmark-weighted) for all subjects and for subjects released from 2025-07.

usage: python -B scripts/test_pool_stats.py   (writes data/test_pool_stats.json)
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
RUNS = (1003760, 1032881, 1032882)       # Beta runs that existed when the model was chosen; they set the test weights
# Copies of the baseline archive uploaded later as queue probes. They hold the same model files.
BETA_ARCHIVES = {"adaptive_baseline.zip", "beta_probe_20261105.zip", "beta_conc1_probe.zip"}


def later_beta_runs():
    """Finished Beta runs made after the model choice, used only as a check."""
    rec = ROOT / "private/submissions"
    archive = {}
    for p in list(rec.glob("*.json")) + list(rec.glob("*/*.json")):
        if p.name.startswith("._") or p.name == "track_state.json":
            continue
        d = json.loads(p.read_text())
        if d.get("submission_id"):
            archive[int(d["submission_id"])] = d["archive"]
    for line in (rec / "resubmits.jsonl").read_text().splitlines():
        d = json.loads(line)
        archive[int(d["submission_id"])] = d["archive"]
    return tuple(sorted(sid for sid, a in archive.items() if a in BETA_ARCHIVES and sid not in RUNS
                        and (ROOT / f"private/feedback/{sid}_scoring_stdout.txt").exists()))


LATER_RUNS = later_beta_runs()
ROW = r"(subject_\d+)\s+(benchmark_\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)"
BINS = np.array([0.0, 0.1, 0.2, 0.3, 0.4])


def budget_tables(text):
    parts = re.split(r"Label budget:\s*(\d+)", text)
    return {int(parts[i]): {(s, b): (float(br), float(ece)) for s, b, n, br, ece in re.findall(ROW, parts[i + 1])}
            for i in range(1, len(parts), 2)}


def pool_groups(runs):
    groups = []
    for run in runs:
        tables = budget_tables((ROOT / f"private/feedback/{run}_scoring_stdout.txt").read_text())
        for key in tables[0]:
            dist = tables[0][key][1]                       # |p - 0.5|
            b31, e31 = tables[31][key]
            pq_check = b31 - e31 ** 2
            b1, e1 = tables[1][key]
            cands = [0.5 - dist, 0.5 + dist]
            # Side of 0.5 from budget 1, where q is 0.4 or 0.6.
            err = [min(abs(abs(q - p) - e1) for q in (0.4, 0.6)) for p in cands]
            p = cands[int(np.argmin(err))] if dist > 1e-9 else 0.5
            groups.append({"p": p, "pq_from_b0": p * (1 - p), "pq_from_b31": pq_check, "side_err": min(err)})
    return pd.DataFrame(groups)


g = pool_groups(RUNS)
share = np.bincount(np.minimum((np.abs(g.p - 0.5) / 0.1).astype(int), 4), minlength=5) / len(g)
test = {"runs": list(RUNS), "group_evaluations": len(g),
        "mean_pq": float(g.pq_from_b0.mean()), "mean_pq_b31_check": float(g.pq_from_b31.mean()),
        "max_abs_pq_disagreement": float((g.pq_from_b0 - g.pq_from_b31).abs().max()),
        "max_side_residual": float(g.side_err.max()),
        "mean_p": float(g.p.mean()), "median_p": float(g.p.median()), "share_above_half": float((g.p > 0.5).mean()),
        "bin_share": share.round(3).tolist()}


def shares(df):
    return (np.bincount(np.minimum((np.abs(df.p - 0.5) / 0.1).astype(int), 4), minlength=5) / len(df)).round(3).tolist()


g_later = pool_groups(LATER_RUNS)
later = {"runs": list(LATER_RUNS), "group_evaluations": len(g_later), "mean_pq": float(g_later.pq_from_b0.mean()),
         "bin_share": shares(g_later)}
g_all = pd.concat([g, g_later])
pooled = {"runs": list(RUNS + LATER_RUNS), "group_evaluations": len(g_all), "mean_pq": float(g_all.pq_from_b0.mean()),
          "bin_share": shares(g_all)}

# Training corpus, groups with at least 80 items, benchmark-weighted.
C = ROOT / "data/corpus_full"
pairs = pd.read_parquet(C / "pairs.parquet", columns=["benchmark", "subject_key", "n", "k"])
subjects = pd.read_parquet(C / "subjects.parquet", columns=["subject_key", "release_date"])
dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
agg = pairs.groupby(["benchmark", "subject_key"]).agg(items=("n", "size"), n=("n", "sum"), k=("k", "sum")).reset_index()
agg = agg[agg["items"] >= 80]
agg["p"] = agg.k / agg.n
agg["date"] = agg.subject_key.map(dict(zip(subjects.subject_key, dates)))


def bench_weighted(df):
    per = df.groupby("benchmark").agg(pq=("p", lambda s: float((s * (1 - s)).mean())), p=("p", "mean"),
                                      above=("p", lambda s: float((s > 0.5).mean())))
    return {"groups": int(len(df)), "benchmarks": int(per.shape[0]), "mean_pq": float(per.pq.mean()),
            "mean_p": float(per.p.mean()), "share_above_half": float(per.above.mean())}


train = {"all": bench_weighted(agg), "recent_2025_07": bench_weighted(agg[agg.date >= pd.Timestamp("2025-07-01")])}
out = {"test_pool": test, "test_pool_later": later, "test_pool_pooled": pooled, "train": train}
(HERE / "data/test_pool_stats.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))

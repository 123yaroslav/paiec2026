"""Candidate ladder of the final model in the official-structure simulator.

Uses a frozen copy of colab_jobs/sim_official.py (scripts/sim_official_v2_snapshot.py)
with its date prior, fold-specific name ability and E5 embeddings. For every group it
stores the six-budget Brier scores of each variant and the group accuracy p, so that
test weights (reweighting groups by |p - 0.5| bins to the formative pool) are applied
afterwards in summarize().

usage: python -B scripts/sim_ladder.py --tag recent_dev --min-date 2025-01-01 --seed0 0
       python -B scripts/sim_ladder.py --tag recent_fresh --min-date 2025-01-01 --seed0 200000
       python -B scripts/sim_ladder.py --tag all_dev --seed0 0
Writes data/ladder_<tag>.npz.
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_official_v2_snapshot as so  # noqa: E402

KERNEL = [0.03, 3.0]
WIDE = {"tau_c": 2.5, "tau_d": 1.0}
VARIANTS = {
    "beta": {},
    "L1c": {"tau_c": 1.2, "tau_d": 0.6, "item_exact": 2.0},
    "L2c": {"tau_c": 1.2, "tau_d": 0.6, "item_exact": 2.0, "item": KERNEL, "use_z": True, "gamma": 0.3},
    "wide_level": dict(WIDE),
    "wide_exact": dict(WIDE, item_exact=1.0),
    "wide_kernel": dict(WIDE, item_exact=1.0, item=KERNEL),
    "L3": dict(WIDE, item_exact=1.0, item=KERNEL, use_z=True, gamma=0.3),
    "L4": dict(WIDE, item_exact=1.0, item=KERNEL, use_z=True, gamma=0.3, date_prior=True, date_scale=1.0),
    "L5": dict(WIDE, item_exact=1.0, item=KERNEL, use_z=True, gamma=0.3, date_prior=True, date_scale=1.0,
               name_scale=0.8),
    "L5_no_text": dict(WIDE, item_exact=1.0, item=KERNEL, date_prior=True, date_scale=1.0, name_scale=0.8),
    "L6": {"tau_c": 2.0, "tau_d": 0.8, "item_exact": 1.0, "item": KERNEL, "use_z": True, "gamma": 0.45,
           "date_prior": True, "date_scale": 1.0, "name_scale": 1.2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-date", default="")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    pairs = pd.read_parquet(ROOT / "colab_jobs/results/exportB/work/results/pairs_z.parquet")
    subjects = pd.read_parquet(ROOT / "data/corpus_full/subjects.parquet",
                               columns=["subject_key", "release_date", "normalized_name"])
    dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
    dp = json.loads((ROOT / "colab_jobs/results/date_prior.json").read_text())
    yrs = ((dates - pd.Timestamp("2023-01-01")).dt.days / 365.25).clip(*dp["clip"])
    offset = pd.Series(np.where(dates.notna(), dp["a"] + dp["b"] * yrs.fillna(0), dp["a0"]), index=subjects.subject_key)
    if args.min_date:
        pairs = pairs[pairs.subject_key.isin(set(subjects.subject_key[dates >= pd.Timestamp(args.min_date)]))]
    state = {"index": so.build_index(pairs), "variants": VARIANTS,
             "names": dict(zip(subjects.subject_key, subjects.normalized_name.fillna("").str.lower().str.strip())),
             "name_ab": json.loads((ROOT / "colab_jobs/results/name_ability_folds.json").read_text()),
             "date_offset": offset.to_dict(), "date_a0": dp["a0"],
             "emb": np.load(ROOT / "data/corpus_full/item_emb.npy", mmap_mode="r")}
    print(args.tag, "benchmarks", len(state["index"]), "with 2+ subjects",
          sum(len(v) >= 2 for v in state["index"].values()), flush=True)
    seeds = range(args.seed0, args.seed0 + args.draws)
    with mp.get_context("fork").Pool(args.procs, initializer=so.init, initargs=(state,)) as pool:
        results = pool.map(so.run_draw, seeds)
    draw_id = np.concatenate([np.full(len(r["_p"]), k) for k, r in enumerate(results)])
    rows = {name: np.array([row for r in results for row in r[name]]) for name in VARIANTS}
    np.savez(HERE / f"data/ladder_{args.tag}.npz", draw_id=draw_id, p=np.array([p for r in results for p in r["_p"]]),
             seed0=args.seed0, min_date=args.min_date, n_benchmarks=len(state["index"]),
             variants=json.dumps(VARIANTS), **rows)
    print("wrote", args.tag, "groups", len(draw_id), flush=True)


if __name__ == "__main__":
    main()

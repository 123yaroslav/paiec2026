"""Stronger baselines and a no-sharing ablation on the ladder's confirmation draws.

Uses the same frozen simulator copy and the same draws as sim_ladder.py (a draw
depends only on its seed and the pair index). Adds
  beta_a<a>   Beta-binomial with a Beta(a, a) prior, (k + a) / (m + 2a)
  *_own       a model variant that sees only the labels of the target group,
              so nothing is shared across subjects (no common benchmark offset,
              no same-item shift, similar-item shift from own labels only)
and repeats beta and L5 so that every difference is paired within this file.

usage: python -B scripts/sim_baselines.py --tag recent_conf --min-date 2025-01-01 --seed0 300000
       python -B scripts/sim_baselines.py --tag all_conf --seed0 300000
       python -B scripts/sim_baselines.py --tag recent_conf --min-date 2025-01-01 --only beta,L3,L3_own --out baselines_l3
Writes data/<out>_<tag>.npz (default out: baselines).
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
L5 = dict(WIDE, item_exact=1.0, item=KERNEL, use_z=True, gamma=0.3, date_prior=True, date_scale=1.0, name_scale=0.8)
L3 = dict(WIDE, item_exact=1.0, item=KERNEL, use_z=True, gamma=0.3)
VARIANTS = {
    "beta": {},
    "beta_a0.25": {"beta_a": 0.25},
    "beta_a0.5": {"beta_a": 0.5},
    "beta_a2": {"beta_a": 2.0},
    "wide_level": dict(WIDE),
    "wide_level_own": dict(WIDE, own=True),
    "L5": dict(L5),
    "L5_own": dict(L5, own=True),
    "L3": dict(L3),
    "L3_own": dict(L3, own=True),
}


def run_draw(seed):
    rng = np.random.default_rng(seed)
    draw = so.make_draw(rng)
    out = {"_p": [float((g["acq"].k.sum() + g["eval"].k.sum()) / (g["acq"].n.sum() + g["eval"].n.sum())) for g in draw]}
    for name, v in so.G["variants"].items():
        rows = []
        for gi, g in enumerate(draw):
            scores = []
            for b in so.BUDGETS:
                if name == "beta" or "beta_a" in v:
                    a = v.get("beta_a", 1.0)
                    m = min(b, len(g["acq"]))
                    p = np.full(len(g["eval"]), (g["y"][:m].sum() + a) / (m + 2 * a))
                elif v.get("own"):
                    p = so.predict_group([g], 0, b, v)
                else:
                    p = so.predict_group(draw, gi, b, v)
                scores.append(so.brier(p, g["eval"]))
            rows.append(scores)
        out[name] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-date", default="")
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated variant names, e.g. beta,L3,L3_own")
    ap.add_argument("--out", default="baselines", help="output file prefix in data/")
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
    variants = {k: v for k, v in VARIANTS.items() if not args.only or k in args.only.split(",")}
    state = {"index": so.build_index(pairs), "variants": variants,
             "names": dict(zip(subjects.subject_key, subjects.normalized_name.fillna("").str.lower().str.strip())),
             "name_ab": json.loads((ROOT / "colab_jobs/results/name_ability_folds.json").read_text()),
             "date_offset": offset.to_dict(), "date_a0": dp["a0"],
             "emb": np.load(ROOT / "data/corpus_full/item_emb.npy", mmap_mode="r")}
    print(args.tag, "benchmarks", len(state["index"]), flush=True)
    seeds = range(args.seed0, args.seed0 + args.draws)
    with mp.get_context("fork").Pool(args.procs, initializer=so.init, initargs=(state,)) as pool:
        results = pool.map(run_draw, seeds)
    draw_id = np.concatenate([np.full(len(r["_p"]), k) for k, r in enumerate(results)])
    rows = {name: np.array([row for r in results for row in r[name]]) for name in variants}
    np.savez(HERE / f"data/{args.out}_{args.tag}.npz", draw_id=draw_id, p=np.array([p for r in results for p in r["_p"]]),
             seed0=args.seed0, min_date=args.min_date, n_benchmarks=len(state["index"]),
             variants=json.dumps(variants), **rows)
    print("wrote", args.tag, "groups", len(draw_id), flush=True)


if __name__ == "__main__":
    main()

"""Simulated official protocol with judge ratings as the item logit.

The base logit of an item is -beta * standardized rating, so harder items get
lower success probability. Levels adapt exactly as in train_eval3. Pairs are
restricted to rated items. Benchmarks form 5 folds only for reporting, the
judge has no fitted parameters apart from the grid.

usage: sim_judge.py RATINGS_PARQUET [--draws N]
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("RUNTIME_DIR", PAIEC_ROOT + "/v5")
import train_eval3 as te  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ratings")
ap.add_argument("--draws", type=int, default=150)
ap.add_argument("--pairs", default=PAIEC_ROOT + "/data/corpus_full/pairs.parquet")
args = ap.parse_args()

ratings = pd.read_parquet(args.ratings)
mu, sd = ratings.rating.mean(), ratings.rating.std()
rstd = pd.Series(((ratings.rating - mu) / sd).to_numpy(), index=ratings.item_key)
pairs = pd.read_parquet(args.pairs)
pairs = pairs[pairs.item_key.isin(set(ratings.item_key))].copy()
pairs["s"] = pd.factorize(pairs.subject_key)[0]
pairs["r"] = rstd.reindex(pairs.item_key).to_numpy()
rng = np.random.default_rng(0)
groups = te.make_groups(pairs, rng, min_items=80, max_items=120)
for g in groups:
    g["r_acq"] = g["acq"].r.to_numpy()
    g["r_eval"] = g["eval"].r.to_numpy()
draws = te.sample_draws(groups, rng, args.draws)
print("groups", len(groups), "benchmarks", len({g["bench"] for g in groups}), "draws", len(draws), flush=True)

base = te.simulate_baseline(groups, draws, "beta")
print("BETA", round(base[0], 5), np.round(base[1], 4).tolist(), flush=True)
results = []
for beta, tau_c, tau_d in itertools.product((0.0, 0.15, 0.3, 0.5, 0.8), (1.2, 1.8, 2.5), (0.8, 1.2)):
    for g in groups:
        g["z_acq"] = -beta * g["r_acq"]
        g["z_eval"] = -beta * g["r_eval"]
    alc, pb = te.simulate(groups, draws, None, (tau_c, tau_d, 1.0))
    results.append((alc, beta, tau_c, tau_d, pb))
    print(f"beta {beta:.2f} tau_c {tau_c} tau_d {tau_d} ALC {alc:.5f}", np.round(pb, 4).tolist(), flush=True)
best = min(results)
print("BEST", round(best[0], 5), "beta", best[1], "tau_c", best[2], "tau_d", best[3], np.round(best[4], 4).tolist())
zero = min(r for r in results if r[1] == 0.0)
print("BEST without judge", round(zero[0], 5), "tau_c", zero[2], "tau_d", zero[3])

"""Within-benchmark signal of judge ratings against observed item accuracy.

usage: judge_analysis.py RATINGS_PARQUET [PAIRS_PARQUET]
Prints per-benchmark Spearman correlation between the rating and the item
accuracy averaged over subjects, then the pooled summary.
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import sys

import numpy as np
import pandas as pd

ratings = pd.read_parquet(sys.argv[1])
pairs = pd.read_parquet(sys.argv[2] if len(sys.argv) > 2 else PAIEC_ROOT + "/data/corpus_full/pairs.parquet",
                        columns=["benchmark", "item_key", "n", "k"])
acc = pairs[pairs.item_key.isin(set(ratings.item_key))].groupby("item_key").agg(n=("n", "sum"), k=("k", "sum"))
acc["acc"] = acc.k / acc.n
d = ratings.merge(acc, left_on="item_key", right_index=True)
rows = []
for bench, g in d.groupby("benchmark"):
    if len(g) < 30 or g.acc.std() < 1e-6 or g.rating.std() < 1e-6:
        continue
    rho = g[["rating", "acc"]].rank().corr().iloc[0, 1]
    rows.append((bench, len(g), rho, g.acc.mean(), g.rating.mean()))
r = pd.DataFrame(rows, columns=["benchmark", "items", "spearman", "mean_acc", "mean_rating"])
print(r.sort_values("spearman").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print("benchmarks", len(r), "median spearman", round(r.spearman.median(), 3), "mean", round(r.spearman.mean(), 3),
      "share negative", round((r.spearman < 0).mean(), 3))
between = d.groupby("benchmark").agg(acc=("acc", "mean"), rating=("rating", "mean"))
print("between-benchmark spearman of mean rating and mean accuracy", round(between.rank().corr().iloc[0, 1], 3))

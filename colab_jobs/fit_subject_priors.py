"""Fit the release-date prior and the model-name ability table from training groups.

Outputs
  colab_jobs/results/date_prior.json        logit accuracy = a + b * years since 2023 (clipped), a0 if no date
  v5/name_ability.json                       name -> shrunk mean within-benchmark skill, all benchmarks
  colab_jobs/results/name_ability_folds.json fold -> name -> skill from the other folds (simulator only)

Groups are subject-benchmark pairs. The date prior uses groups with at least 80
items and weights every benchmark equally. Name skill is the group logit minus
the benchmark mean logit, averaged per normalized_name with pseudo-count 2.
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("PAIEC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = f"{ROOT}/data/corpus_full"


def groups(min_items):
    p = pd.read_parquet(f"{C}/pairs.parquet", columns=["benchmark", "subject_key", "n", "k"])
    s = pd.read_parquet(f"{C}/subjects.parquet", columns=["subject_key", "release_date", "normalized_name"])
    g = p.groupby(["benchmark", "subject_key"]).agg(items=("n", "size"), k=("k", "sum"), n=("n", "sum")).reset_index()
    g = g[g["items"] >= min_items].merge(s, on="subject_key", how="left")
    g["y"] = np.log((g.k + 0.5) / (g.n - g.k + 0.5))
    return g


def date_prior():
    g = groups(80)
    g["date"] = pd.to_datetime(g.release_date.str.slice(0, 10), errors="coerce")
    g["yrs"] = ((g.date - pd.Timestamp("2023-01-01")).dt.days / 365.25).clip(-3, 4)
    w = 1 / g.groupby("benchmark").benchmark.transform("size")
    known = g.date.notna()
    X = np.stack([np.ones(known.sum()), g.yrs[known]], 1)
    W = w[known].to_numpy()
    coef = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * g.y[known].to_numpy()))
    a0 = float(np.average(g.y[~known], weights=w[~known]))
    return {"a": float(coef[0]), "b": float(coef[1]), "a0": a0, "clip": [-3, 4]}


def name_table(g):
    g = g.copy()
    g["rel"] = g.y - g.groupby("benchmark").y.transform("mean")
    g["name"] = g.normalized_name.fillna("").str.lower().str.strip()
    g = g[g.name != ""]
    agg = g.groupby("name").rel.agg(["sum", "count"])
    return (agg["sum"] / (agg["count"] + 2.0)).round(4).to_dict()


def main():
    json.dump(date_prior(), open(f"{ROOT}/colab_jobs/results/date_prior.json", "w"))
    g = groups(20)
    json.dump(name_table(g), open(f"{ROOT}/v5/name_ability.json", "w"))
    # Folds are the benchmark folds of export_z.py, read from its held-out logits.
    folds = pd.read_parquet(f"{ROOT}/colab_jobs/results/exportB/work/results/pairs_z.parquet",
                            columns=["benchmark", "fold"]).drop_duplicates("benchmark")
    g = g.merge(folds, on="benchmark")
    out = {str(int(f)): name_table(g[g.fold != f]) for f in sorted(g.fold.unique())}
    json.dump(out, open(f"{ROOT}/colab_jobs/results/name_ability_folds.json", "w"))
    print("date prior", json.load(open(f"{ROOT}/colab_jobs/results/date_prior.json")))


if __name__ == "__main__":
    main()

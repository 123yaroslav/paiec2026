"""Judge, name-ability and runtime-replay statistics from saved files.

usage: python -B scripts/extra_stats.py   (writes data/extra_stats.json)
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
R = ROOT / "colab_jobs/results"
C = ROOT / "data/corpus_full"
OUT = {}
TEST_SHARE_SIM = np.array(json.loads((HERE / "data/test_pool_stats.json").read_text())["test_pool"]["bin_share"])  # recomputed shares


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


# ------------------------------------------------------------ judge
ratings = pd.read_parquet(R / "judge_partial/qwen3_8b_partial.parquet")
rated = ratings[ratings.rating.notna()]
pairs = pd.read_parquet(C / "pairs.parquet", columns=["benchmark", "subject_key", "item_key", "n", "k"])
acc = pairs[pairs.item_key.isin(set(rated.item_key))].groupby("item_key").agg(n=("n", "sum"), k=("k", "sum"))
acc["acc"] = acc.k / acc.n
d = rated.merge(acc, left_on="item_key", right_index=True)
rows = []
for bench, g in d.groupby("benchmark"):
    if len(g) < 30 or g.acc.std() < 1e-6 or g.rating.std() < 1e-6:
        continue
    rows.append(spearman(g.rating, g.acc))
between = d.groupby("benchmark").agg(acc=("acc", "mean"), rating=("rating", "mean"))
OUT["judge"] = {"sampled": int(len(ratings)), "rated": int(len(rated)), "benchmarks": len(rows),
                "median_within": float(np.median(rows)), "share_negative": float(np.mean(np.array(rows) < 0)),
                "between": spearman(between.rating, between.acc)}

# ------------------------------------------------------------ name ability, held-out within-benchmark signal
folds = json.loads((R / "name_ability_folds.json").read_text())
pz = pd.read_parquet(R / "exportB/work/results/pairs_z.parquet", columns=["benchmark", "subject_key", "n", "k", "fold"])
subjects = pd.read_parquet(C / "subjects.parquet", columns=["subject_key", "normalized_name"])
name_of = dict(zip(subjects.subject_key, subjects.normalized_name.fillna("").str.lower().str.strip()))
grp = pz.groupby(["benchmark", "subject_key"]).agg(items=("n", "size"), n=("n", "sum"), k=("k", "sum"),
                                                     fold=("fold", "first")).reset_index()
grp = grp[grp["items"] >= 80]
grp["logit"] = np.log((grp.k + 0.5) / (grp.n - grp.k + 0.5))
grp["ability"] = [folds[str(f)].get(name_of.get(s, ""), np.nan) for f, s in zip(grp.fold, grp.subject_key)]
rho = []
for bench, g in grp.dropna(subset=["ability"]).groupby("benchmark"):
    if len(g) >= 5 and g.ability.std() > 0 and g.logit.std() > 0:
        r = spearman(g.ability, g.logit)
        if np.isfinite(r):
            rho.append(r)
OUT["name_ability"] = {"names_runtime": len(json.loads((ROOT / "v5/name_ability.json").read_text())),
                       "benchmarks": len(rho), "share_positive": float(np.mean(np.array(rho) > 0)),
                       "median": float(np.median(rho)), "min_groups": 5}

# ------------------------------------------------------------ runtime replays of V2 and V3
dump = pd.read_parquet(R / "sim_dump_recent_ex.parquet")
bins = np.minimum((np.abs(dump.p - 0.5) / 0.1).astype(int), 4).to_numpy()
train_share = np.bincount(bins, minlength=5) / len(bins)
dump["w"] = TEST_SHARE_SIM[bins] / np.maximum(train_share[bins], 1e-9)
dump["w"] /= dump.w.mean()


def ratio_se(df, col):
    """Weighted mean of col over groups and its standard error with draws as clusters."""
    tot = df.assign(wd=df.w * df[col]).groupby("seed").agg(W=("w", "sum"), D=("wd", "sum"))
    r = tot.D.sum() / tot.W.sum()
    k = len(tot)
    se = math.sqrt(((tot.D - r * tot.W) ** 2).sum() * k / (k - 1)) / tot.W.sum()
    return float(r), float(se)


OUT["replay"] = {}
for name in ("v2", "v3"):
    rep = pd.read_parquet(R / f"replay_{name}.parquet")[["seed", "group", "alc"]].rename(columns={"alc": "run"})
    m = dump.merge(rep, on=["seed", "group"])
    out = {"draws": int(m.seed.nunique()), "groups": int(len(m))}
    for label, df in (("testw", m), ("unweighted", m.assign(w=1.0))):
        for col in ("run", "L5", "beta"):
            out[f"{label}_{col}"] = ratio_se(df, col)[0]
        out[f"{label}_L5_minus_run"], out[f"{label}_L5_minus_run_se"] = ratio_se(df.assign(dd=df.L5 - df.run), "dd")
        out[f"{label}_run_minus_beta"], out[f"{label}_run_minus_beta_se"] = ratio_se(df.assign(dd=df.run - df.beta), "dd")
    OUT["replay"][name] = out
chk = pd.read_parquet(R / "replay_L5_check.parquet")[["seed", "group", "alc"]].merge(dump, on=["seed", "group"])
OUT["replay"]["L5_check"] = {"draws": int(chk.seed.nunique()), "groups": int(len(chk)),
                             "mean_abs_diff": float((chk.alc - chk.L5).abs().mean()),
                             "runtime_mean": float(chk.alc.mean()), "sim_mean": float(chk.L5.mean())}
(HERE / "data/extra_stats.json").write_text(json.dumps(OUT, indent=1) + "\n")
print(json.dumps(OUT, indent=1))

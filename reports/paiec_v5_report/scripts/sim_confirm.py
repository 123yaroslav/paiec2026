"""Confirmation run of the frozen V5 configuration on fresh simulator draws.

Reuses make_draw, predict_group and brier from colab_jobs/sim_official.py
unchanged. Hyperparameters are fixed to the tuned values (tau_c 1.2,
tau_d 0.6, sigma_exact^2 2, kernel bandwidth 0.03, sigma_k^2 3, gamma 0.3)
and components are added one at a time. Draw seeds start at SEED0, so they do
not overlap the seeds 0..599 used for development and tuning.

Writes data/confirm_rows.npz (per group Brier at six budgets for every
variant) and data/confirm_summary.json (mean ALC, paired difference with the
Beta(1,1) baseline and its standard error over groups and over draws).

usage: python -B scripts/sim_confirm.py [--draws 600] [--seed0 100000] [--procs 6]
       python -B scripts/sim_confirm.py --check-dev 600   (Beta only, seeds 0..N-1)
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(PAIEC_ROOT + "")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_official_snapshot as so  # noqa: E402

PAIRS_Z = ROOT / "colab_jobs/results/exportB/work/results/pairs_z.parquet"
EMB = ROOT / "data/corpus_full/item_emb.npy"
LEVEL = {"tau_c": 1.2, "tau_d": 0.6}
VARIANTS = {
    "level": dict(LEVEL),
    "level_exact": dict(LEVEL, item_exact=2.0),
    "level_exact_kernel": dict(LEVEL, item_exact=2.0, item=[0.03, 3.0]),
    "full": dict(LEVEL, item_exact=2.0, item=[0.03, 3.0], use_z=True, gamma=0.3),
    "prior_only": dict(LEVEL, use_z=True, gamma=0.3),
    "full_gamma1": dict(LEVEL, item_exact=2.0, item=[0.03, 3.0], use_z=True, gamma=1.0),
}


def baseline_rows(draw, a, b):
    rows = []
    for g in draw:
        scores = []
        for budget in so.BUDGETS:
            m = min(budget, len(g["acq"]))
            p = np.full(len(g["eval"]), (g["y"][:m].sum() + a) / (m + a + b))
            scores.append(so.brier(p, g["eval"]))
        rows.append(scores)
    return rows


def run_draw(seed):
    rng = np.random.default_rng(seed)
    draw = so.make_draw(rng)
    out = {"beta11": baseline_rows(draw, 1.0, 1.0), "beta22": baseline_rows(draw, 2.0, 2.0),
           "n_groups": len(draw), "n_bench": len({g["bench"] for g in draw})}
    for name, v in so.G["variants"].items():
        out[name] = [[so.brier(so.predict_group(draw, gi, b, v), g["eval"]) for b in so.BUDGETS]
                     for gi, g in enumerate(draw)]
    return out


def run_beta_only(seed):
    rng = np.random.default_rng(seed)
    return baseline_rows(so.make_draw(rng), 1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--check-dev", type=int, default=0)
    args = ap.parse_args()
    pairs = pd.read_parquet(PAIRS_Z)
    variants = {} if args.check_dev else VARIANTS
    state = {"index": so.build_index(pairs), "variants": variants, "emb": np.load(EMB, mmap_mode="r")}
    print("benchmarks", len(state["index"]), "with 2+ subjects",
          sum(len(v) >= 2 for v in state["index"].values()), flush=True)
    with mp.get_context("fork").Pool(args.procs, initializer=so.init, initargs=(state,)) as pool:
        if args.check_dev:
            res = pool.map(run_beta_only, range(args.check_dev))
            rows = np.array([r for d in res for r in d]) @ so.WEIGHTS
            print(f"Beta(1,1) on dev seeds 0..{args.check_dev - 1}: ALC {rows.mean():.5f}")
            return
        results = pool.map(run_draw, range(args.seed0, args.seed0 + args.draws))
    names = ["beta11", "beta22"] + list(VARIANTS)
    draw_id = np.concatenate([np.full(r["n_groups"], k) for k, r in enumerate(results)])
    rows = {n: np.array([row for r in results for row in r[n]]) for n in names}
    np.savez(HERE / "data/confirm_rows.npz", draw_id=draw_id, budgets=np.array(so.BUDGETS),
             weights=so.WEIGHTS, **rows)
    base = rows["beta11"] @ so.WEIGHTS
    summary = {"draws": args.draws, "seed0": args.seed0, "groups": int(len(draw_id)),
               "groups_per_draw": sorted({r["n_groups"] for r in results}),
               "benchmarks_per_draw": sorted({r["n_bench"] for r in results}), "variants": {}}
    for n in names:
        alc = rows[n] @ so.WEIGHTS
        diff = alc - base
        per_draw = pd.Series(diff).groupby(draw_id).mean().to_numpy()
        summary["variants"][n] = {
            "alc": float(alc.mean()), "se_alc_draws": float(pd.Series(alc).groupby(draw_id).mean().std() / math.sqrt(args.draws)),
            "diff_vs_beta11": float(diff.mean()), "rel_diff_pct": float(100 * diff.mean() / base.mean()),
            "se_diff_groups": float(diff.std(ddof=1) / math.sqrt(len(diff))),
            "se_diff_draws": float(per_draw.std(ddof=1) / math.sqrt(len(per_draw))),
            "win_share_groups": float((diff < 0).mean()),
            "per_budget": rows[n].mean(0).round(5).tolist(),
            "per_budget_se": (rows[n].std(0, ddof=1) / math.sqrt(len(alc))).round(5).tolist(),
            "config": VARIANTS.get(n) or {"beta11": "(k+1)/(m+2)", "beta22": "(k+2)/(m+4)"}[n],
        }
        s = summary["variants"][n]
        print(f"{n:<20} ALC {s['alc']:.5f} diff {s['diff_vs_beta11']:+.5f} ({s['rel_diff_pct']:+.2f}%) "
              f"se_groups {s['se_diff_groups']:.5f} se_draws {s['se_diff_draws']:.5f} per budget {s['per_budget']}",
              flush=True)
    (HERE / "data/confirm_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()

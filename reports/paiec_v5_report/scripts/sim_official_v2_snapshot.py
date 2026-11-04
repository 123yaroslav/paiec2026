"""Simulator with the structure of official formative draws, run on the Mac CPU.

A draw takes 7 held-out benchmarks. Two of them get two subjects, the rest one,
so a draw has 9 groups, as in our formative feedback. Subjects on one benchmark
share the same item set of 88 to 152 items, because the platform takes whole
groups. Each group splits its items 50/50 into acquisition and evaluation
independently, acquisition order is random, and one recorded trial is revealed
per acquired item. Every prediction sees all labels of the draw.

Inputs come from export_z.py (held-out logits z per pair) and the E5 item
embeddings. Variants are listed in VARIANTS and scored on identical draws.
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, PAIEC_ROOT + "/v5")
import adapt  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = np.array((0.1, 0.2, 0.2, 0.2, 0.2, 0.1))
G = {}


def build_index(pairs):
    index = {}
    for bench, gb in pairs.groupby("benchmark", sort=False):
        subs = {}
        for skey, gs in gb.groupby("subject_key", sort=False):
            if len(gs) >= 88:
                subs[skey] = gs.set_index("item_key")
        if subs:
            index[bench] = subs
    return index


def make_draw(rng):
    index = G["index"]
    benches = list(index)
    multi = [b for b in benches if len(index[b]) >= 2]
    for _ in range(50):
        two = list(rng.choice(multi, size=min(2, len(multi)), replace=False))
        rest = [b for b in benches if b not in two]
        one = list(rng.choice(rest, size=5, replace=False))
        groups, total = [], 0
        ok = True
        for b, count in [(b, 2) for b in two] + [(b, 1) for b in one]:
            subs = list(index[b])
            chosen = list(rng.choice(subs, size=count, replace=False))
            common = set(index[b][chosen[0]].index)
            for s in chosen[1:]:
                common &= set(index[b][s].index)
            if len(common) < 88:
                ok = False
                break
            m = int(rng.integers(88, 153))
            items = list(rng.choice(sorted(common), size=min(m, len(common)), replace=False))
            if total + len(items) * count > 1000:
                continue
            total += len(items) * count
            for s in chosen:
                g = index[b][s].loc[items]
                order = rng.permutation(len(g))
                n_acq = (len(g) + 1) // 2
                acq, ev = g.iloc[order[:n_acq]], g.iloc[order[n_acq:]]
                y = (rng.random(len(acq)) < (acq.k / acq.n).to_numpy()).astype(float)
                groups.append({"bench": b, "s": s, "acq": acq, "eval": ev, "y": y})
        if ok and len(groups) >= 7:
            return groups
    raise RuntimeError("could not build a draw")


def brier(p, ev):
    k, n = ev.k.to_numpy(float), ev.n.to_numpy(float)
    return float((k * (1 - p) ** 2 + (n - k) * p ** 2).sum() / n.sum())


def zcol(g, part, v):
    df = g[part]
    z = v.get("gamma", 1.0) * df.z.to_numpy() if v.get("use_z") else np.zeros(len(df))
    if v.get("date_prior"):
        z = z + v.get("date_scale", 1.0) * G["date_offset"].get(g["s"], G["date_a0"])
    if v.get("name_scale"):
        fold = int(df.fold.iloc[0])
        z = z + v["name_scale"] * max(-2.5, min(2.5, G["name_ab"][str(fold)].get(G["names"].get(g["s"], ""), 0.0)))
    if v.get("judge_beta"):
        z = z - v["judge_beta"] * df.r.to_numpy()
    return z


def predict_group(draw, gi, budget, v):
    g = draw[gi]
    zs, ss, ys, keys, idx, ikeys = [], [], [], [], [], []
    for h in draw:
        if h["bench"] != g["bench"]:
            continue
        m = min(budget, len(h["acq"]))
        if m:
            zs.append(zcol(h, "acq", v)[:m])
            ss.append(np.full(m, hash(h["s"]) % (1 << 30)))
            ys.append(h["y"][:m])
            idx.append(h["acq"].i.to_numpy()[:m])
            ikeys.append(h["acq"].index.to_numpy()[:m])
    target = hash(g["s"]) % (1 << 30)
    z_eval = zcol(g, "eval", v)
    if not ys:
        mean, var = adapt.posterior(np.zeros(0), np.zeros(0, int), np.zeros(0), target, v["tau_c"], v["tau_d"], 1.0)
        if v.get("b0_var") is not None:
            var = v["b0_var"]
        return adapt.predict(z_eval, mean, var, 1.0)
    z, s, y = np.concatenate(zs), np.concatenate(ss), np.concatenate(ys)
    mean, var = adapt.posterior(z, s, y, target, v["tau_c"], v["tau_d"], 1.0)
    if v.get("item") or v.get("item_exact"):
        lab_p = np.empty(len(y))
        for sj in np.unique(s):
            msk = s == sj
            mj, _ = adapt.posterior(z, s, y, int(sj), v["tau_c"], v["tau_d"], 1.0)
            lab_p[msk] = 1 / (1 + np.exp(-(z[msk] + mj)))
        resid, w = y - lab_p, lab_p * (1 - lab_p)
        if v.get("item_exact"):
            # Same item labeled for another subject or the same one: identity kernel.
            lab_keys = np.concatenate(ikeys)
            sig2 = v["item_exact"]
            eval_keys = g["eval"].index.to_numpy()
            pos = {}
            for j, kk in enumerate(lab_keys):
                pos.setdefault(kk, []).append(j)
            shift = np.zeros(len(eval_keys))
            for t, kk in enumerate(eval_keys):
                js = pos.get(kk)
                if js:
                    shift[t] = sig2 * resid[js].sum() / (1 + sig2 * w[js].sum())
            z_eval = z_eval + shift
        if v.get("item"):
            bw, sig2 = v["item"]
            x_lab = np.asarray(G["emb"][np.concatenate(idx)], dtype=np.float32)
            x_t = np.asarray(G["emb"][g["eval"].i.to_numpy()], dtype=np.float32)
            if v.get("center"):
                # Remove the shared benchmark template, then renormalize.
                mu = x_lab.mean(0, keepdims=True)
                x_lab = x_lab - mu
                x_t = x_t - mu
                x_lab /= np.linalg.norm(x_lab, axis=1, keepdims=True) + 1e-6
                x_t /= np.linalg.norm(x_t, axis=1, keepdims=True) + 1e-6
            z_eval = z_eval + adapt.item_shift(x_t, x_lab, resid, w, bw, sig2)
    return adapt.predict(z_eval, mean, var, 1.0)


def run_draw(seed):
    rng = np.random.default_rng(seed)
    draw = make_draw(rng)
    out = {"_p": [float((g["acq"].k.sum() + g["eval"].k.sum()) / (g["acq"].n.sum() + g["eval"].n.sum())) for g in draw]}
    for name, v in G["variants"].items():
        rows = []
        for gi, g in enumerate(draw):
            scores = []
            for b in BUDGETS:
                if name == "beta":
                    m = min(b, len(g["acq"]))
                    p = np.full(len(g["eval"]), (g["y"][:m].sum() + 1) / (m + 2))
                else:
                    p = predict_group(draw, gi, b, v)
                scores.append(brier(p, g["eval"]))
            rows.append(scores)
        out[name] = rows
    return out


def init(state):
    G.update(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-z", default=PAIEC_ROOT + "/colab_jobs/results/exportB/work/results/pairs_z.parquet")
    ap.add_argument("--emb", default="")
    ap.add_argument("--ratings", default="")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--variants", required=True, help="JSON dict name -> settings")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-date", default="")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--exclude", default="", help="comma-separated benchmarks to drop")
    ap.add_argument("--dump", default="", help="write per-group ALC rows to this parquet")
    ap.add_argument("--date-prior", default=PAIEC_ROOT + "/colab_jobs/results/date_prior.json")
    args = ap.parse_args()
    pairs = pd.read_parquet(args.pairs_z)
    if args.ratings:
        r = pd.read_parquet(args.ratings)
        rstd = pd.Series(((r.rating - r.rating.mean()) / r.rating.std()).to_numpy(), index=r.item_key)
        pairs = pairs[pairs.item_key.isin(set(r.item_key))].copy()
        pairs["r"] = rstd.reindex(pairs.item_key).to_numpy()
    subjects = pd.read_parquet(PAIEC_ROOT + "/data/corpus_full/subjects.parquet", columns=["subject_key", "release_date"])
    dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
    dp = json.load(open(args.date_prior))
    yrs = ((dates - pd.Timestamp("2023-01-01")).dt.days / 365.25).clip(*dp["clip"])
    offset = pd.Series(np.where(dates.notna(), dp["a"] + dp["b"] * yrs.fillna(0), dp["a0"]), index=subjects.subject_key)
    if args.min_date:
        recent = set(subjects.subject_key[dates >= pd.Timestamp(args.min_date)])
        pairs = pairs[pairs.subject_key.isin(recent)]
    if args.exclude:
        pairs = pairs[~pairs.benchmark.isin(set(args.exclude.split(",")))]
    names = pd.read_parquet(PAIEC_ROOT + "/data/corpus_full/subjects.parquet", columns=["subject_key", "normalized_name"])
    name_of = dict(zip(names.subject_key, names.normalized_name.fillna("").str.lower().str.strip()))
    name_ab = json.load(open(PAIEC_ROOT + "/colab_jobs/results/name_ability_folds.json"))
    state = {"index": build_index(pairs), "variants": json.loads(args.variants), "names": name_of, "name_ab": name_ab,
             "date_offset": offset.to_dict(), "date_a0": dp["a0"],
             "emb": np.load(args.emb, mmap_mode="r") if args.emb else None}
    print("benchmarks", len(state["index"]), "with 2+ subjects", sum(len(v) >= 2 for v in state["index"].values()), flush=True)
    with mp.get_context("fork").Pool(os.cpu_count(), initializer=init, initargs=(state,)) as pool:
        results = pool.map(run_draw, range(args.seed_offset, args.seed_offset + args.draws))
    # Test-pool weights: share of groups by |p - 0.5| estimated from Beta formative feedback (26 groups).
    test_share = np.array([0.115, 0.077, 0.192, 0.308, 0.308])
    gp = np.array([p for res in results for p in res["_p"]])
    bins = np.minimum((np.abs(gp - 0.5) / 0.1).astype(int), 4)
    train_share = np.bincount(bins, minlength=5) / len(bins)
    gw = test_share[bins] / np.maximum(train_share[bins], 1e-9)
    gw = gw / gw.mean()
    print("train share by |p-0.5| bin", np.round(train_share, 3).tolist(), "test share", test_share.tolist(), flush=True)
    summary = {}
    for name in state["variants"]:
        rows = np.array([r for res in results for r in res[name]])
        alc = rows @ WEIGHTS
        summary[name] = {"alc": float(alc.mean()), "alc_testw": float(np.average(alc, weights=gw)),
                         "per_budget": rows.mean(0).round(4).tolist(),
                         "per_budget_testw": np.average(rows, axis=0, weights=gw).round(4).tolist()}
    base = np.array([r for res in results for r in res["beta"]]) @ WEIGHTS if "beta" in summary else None
    for name, s in sorted(summary.items(), key=lambda kv: kv[1]["alc_testw"]):
        rows = np.array([r for res in results for r in res[name]]) @ WEIGHTS
        diff = "" if base is None else (f"vs beta {np.mean(rows - base):+.5f} (se {np.std(rows - base) / math.sqrt(len(rows)):.5f})"
                                         f" testw {np.average(rows - base, weights=gw):+.5f}")
        print(f"{name:<28} ALC {s['alc']:.5f} testw {s['alc_testw']:.5f} {diff} per budget testw {s['per_budget_testw']}", flush=True)
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)
    if args.dump:
        rows = []
        for seed, res in enumerate(results):
            for gi, p in enumerate(res["_p"]):
                row = {"seed": seed, "group": gi, "p": p}
                for name in state["variants"]:
                    row[name] = float(np.dot(res[name][gi], WEIGHTS))
                rows.append(row)
        pd.DataFrame(rows).to_parquet(args.dump)


if __name__ == "__main__":
    main()

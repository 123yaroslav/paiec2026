"""Train the V5 prior on the full corpus and score it with a simulated official protocol.

cv     benchmark folds. Train on the other folds, simulate draws on the held-out fold.
final  train on every benchmark and export weights for the submission.

Simulation follows the starting kit. A draw holds about nine subject-benchmark
groups of 80 to 160 items. Items split 50/50 into acquisition and evaluation,
acquisition order is random, labels are revealed at budgets 0, 1, 3, 7, 15, 31
and every label of every group in the draw is visible to every prediction.
Brier is the mean over all recorded trials of a group, ALC uses the official
weights, and the score is the mean over groups.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.environ.get("RUNTIME_DIR", "/content/run"))
import adapt  # noqa: E402

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
T0 = time.time()


def log(*parts):
    print(f"[{time.time() - T0:7.0f}s]", *parts, flush=True)


# ---------------------------------------------------------------- data

def subject_numeric(subjects):
    dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
    years = ((dates - pd.Timestamp("2023-01-01")).dt.days / 365.25).fillna(0.0).clip(-5, 5)
    missing = dates.isna().astype(float)
    return np.stack([years.to_numpy(), missing.to_numpy()], axis=1).astype(np.float32)


def load(work):
    pairs = pd.read_parquet(f"{work}/pairs.parquet")
    items = pd.read_parquet(f"{work}/items.parquet", columns=["item_key", "benchmark", "text_chars"])
    subjects = pd.read_parquet(f"{work}/subjects.parquet")
    item_emb = np.load(f"{work}/item_emb.npy")
    subj_emb = np.load(f"{work}/subject_emb.npy")
    assert len(item_emb) == len(items) and len(subj_emb) == len(subjects)
    item_idx = pd.Series(np.arange(len(items)), index=items.item_key)
    subj_idx = pd.Series(np.arange(len(subjects)), index=subjects.subject_key)
    pairs["i"] = item_idx.reindex(pairs.item_key).to_numpy()
    pairs["s"] = subj_idx.reindex(pairs.subject_key).to_numpy()
    pairs = pairs.dropna(subset=["i", "s"])
    pairs["i"] = pairs.i.astype(np.int64)
    pairs["s"] = pairs.s.astype(np.int64)
    xi = np.concatenate([item_emb.astype(np.float32),
                         (np.log1p(items.text_chars.to_numpy(np.float32)) / 10)[:, None]], axis=1)
    xs = np.concatenate([subj_emb.astype(np.float32), subject_numeric(subjects)], axis=1)
    log("loaded pairs", len(pairs), "items", len(items), "subjects", len(subjects),
        "benchmarks", pairs.benchmark.nunique())
    return pairs, xi, xs


# ---------------------------------------------------------------- model

class Prior(nn.Module):
    def __init__(self, ds, di, hidden=256, rank=32, dropout=0.1):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(ds, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.i = nn.Sequential(nn.Linear(di, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
        self.a = nn.Linear(hidden, 1)
        self.b = nn.Linear(hidden, 1)
        self.u = nn.Linear(hidden, rank, bias=False)
        self.v = nn.Linear(hidden, rank, bias=False)
        self.rank = rank

    def forward(self, xs, xi):
        hs, hi = self.s(xs), self.i(xi)
        inter = (self.u(hs) * self.v(hi)).sum(-1) / math.sqrt(self.rank)
        return self.a(hs).squeeze(-1) + self.b(hi).squeeze(-1) + inter


def train_prior(pairs, xi, xs, args, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    group_trials = pairs.groupby(["subject_key", "benchmark"]).n.transform("sum").to_numpy(np.float32)
    # Each subject-benchmark group carries equal total weight, as in the official mean.
    w = 1.0 / group_trials
    w = w / w.mean()
    Xi = torch.tensor(xi, device=dev, dtype=torch.float16)
    Xs = torch.tensor(xs, device=dev, dtype=torch.float32)
    I = torch.tensor(pairs.i.to_numpy(), device=dev)
    S = torch.tensor(pairs.s.to_numpy(), device=dev)
    K = torch.tensor(pairs.k.to_numpy(np.float32), device=dev)
    N = torch.tensor(pairs.n.to_numpy(np.float32), device=dev)
    W = torch.tensor(w, device=dev)
    bench_codes, bench_names = pd.factorize(pairs.benchmark)
    B = torch.tensor(bench_codes, device=dev)
    model = Prior(xs.shape[1], xi.shape[1], args.hidden, args.rank, args.dropout).to(dev)
    boff = torch.zeros(len(bench_names), device=dev, requires_grad=True)
    params = [{"params": model.parameters(), "weight_decay": args.wd}]
    if args.bench_offset:
        params.append({"params": [boff], "weight_decay": 0.0})
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * math.ceil(len(pairs) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(pairs), device=dev)
        total = 0.0
        for start in range(0, len(pairs), args.batch):
            b = perm[start:start + args.batch]
            logit = model(Xs[S[b]], Xi[I[b]].float())
            if args.bench_offset:
                logit = logit + boff[B[b]]
            ls = torch.nn.functional.logsigmoid(logit)
            lns = torch.nn.functional.logsigmoid(-logit)
            loss = -(W[b] * (K[b] * ls + (N[b] - K[b]) * lns) / N[b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
            step += 1
        log(f"epoch {epoch + 1}/{args.epochs} loss {total / len(pairs):.4f}")
    model.eval()
    if args.bench_offset:
        with torch.no_grad():
            # A new benchmark gets the average offset, its own offset comes from labels.
            model.a.bias += boff.mean()
        log("benchmark offsets std", float(boff.std()), "mean", float(boff.mean()))
    return model


@torch.no_grad()
def base_logits(model, xs, xi, s_idx, i_idx):
    dev = next(model.parameters()).device
    out = []
    for start in range(0, len(s_idx), 65536):
        s = torch.tensor(xs[s_idx[start:start + 65536]], device=dev)
        i = torch.tensor(xi[i_idx[start:start + 65536]], device=dev, dtype=torch.float32)
        out.append(model(s, i).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------- simulation

def make_groups(pairs, rng, min_items=80, max_items=160):
    """Eligible held-out groups, each subsampled to a draw-like size and split 50/50."""
    groups = []
    for (bench, skey), g in pairs.groupby(["benchmark", "subject_key"], sort=False):
        if len(g) < min_items:
            continue
        g = g.sample(n=min(len(g), int(rng.integers(min_items, max_items + 1))), random_state=int(rng.integers(1 << 30)))
        order = rng.permutation(len(g))
        n_acq = (len(g) + 1) // 2
        acq, ev = g.iloc[order[:n_acq]], g.iloc[order[n_acq:]]
        # One recorded trial is revealed for an acquired item.
        y_acq = (rng.random(len(acq)) < (acq.k / acq.n).to_numpy()).astype(float)
        groups.append({"bench": bench, "s": int(g.s.iloc[0]), "acq": acq, "y_acq": y_acq, "eval": ev})
    return groups


def sample_draws(groups, rng, n_draws, limit=1000):
    by_bench = {}
    for gi, g in enumerate(groups):
        by_bench.setdefault(g["bench"], []).append(gi)
    benches = list(by_bench)
    draws = []
    for _ in range(n_draws):
        used, chosen, count = set(), [], {}
        for _ in range(60):
            # Prefer benchmarks that are less represented in this draw.
            low = min(count.get(b, 0) for b in benches)
            cand = [b for b in benches if count.get(b, 0) == low]
            b = cand[int(rng.integers(len(cand)))]
            options = [gi for gi in by_bench[b] if gi not in chosen]
            if not options:
                count[b] = count.get(b, 0) + 10
                continue
            gi = options[int(rng.integers(len(options)))]
            size = len(groups[gi]["acq"]) + len(groups[gi]["eval"])
            if sum(len(groups[c]["acq"]) + len(groups[c]["eval"]) for c in chosen) + size > limit:
                count[b] = count.get(b, 0) + 10
                if all(count.get(x, 0) >= 10 for x in benches):
                    break
                continue
            chosen.append(gi)
            count[b] = count.get(b, 0) + 1
        draws.append(chosen)
    return draws


def brier(p, ev):
    k, n = ev.k.to_numpy(float), ev.n.to_numpy(float)
    return float((k * (1 - p) ** 2 + (n - k) * p ** 2).sum() / n.sum())


def simulate(groups, draws, zfun, hp):
    """Return mean ALC and mean Brier per budget for one hyperparameter setting."""
    tau_c, tau_d, gamma = hp
    alcs, per_budget = [], np.zeros(len(BUDGETS))
    count = 0
    for draw in draws:
        for gi in draw:
            g = groups[gi]
            scores = []
            for budget in BUDGETS:
                zs, ss, ys = [], [], []
                for gj in draw:
                    h = groups[gj]
                    if h["bench"] != g["bench"]:
                        continue
                    m = min(budget, len(h["acq"]))
                    if m:
                        zs.append(h["z_acq"][:m])
                        ss.append(np.full(m, h["s"]))
                        ys.append(h["y_acq"][:m])
                z = np.concatenate(zs) if zs else np.zeros(0)
                s = np.concatenate(ss) if ss else np.zeros(0, dtype=int)
                y = np.concatenate(ys) if ys else np.zeros(0)
                mean, var = adapt.posterior(z, s, y, g["s"], tau_c, tau_d, gamma)
                p = adapt.predict(g["z_eval"], mean, var, gamma)
                scores.append(brier(p, g["eval"]))
            alcs.append(float(np.dot(WEIGHTS, scores)))
            per_budget += scores
            count += 1
    return float(np.mean(alcs)), (per_budget / max(count, 1)).tolist()


def simulate_baseline(groups, draws, kind):
    alcs, per_budget, count = [], np.zeros(len(BUDGETS)), 0
    for draw in draws:
        for gi in draw:
            g = groups[gi]
            scores = []
            for budget in BUDGETS:
                m = min(budget, len(g["acq"]))
                yk = g["y_acq"][:m].sum()
                p = 0.5 if kind == "const" else (yk + 1) / (m + 2)
                scores.append(brier(np.full(len(g["eval"]), p), g["eval"]))
            alcs.append(float(np.dot(WEIGHTS, scores)))
            per_budget += scores
            count += 1
    return float(np.mean(alcs)), (per_budget / count).tolist()


# ---------------------------------------------------------------- main

def attach_logits(groups, model, xs, xi):
    for g in groups:
        g["z_acq"] = base_logits(model, xs, xi, g["acq"].s.to_numpy(), g["acq"].i.to_numpy())
        g["z_eval"] = base_logits(model, xs, xi, g["eval"].s.to_numpy(), g["eval"].i.to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cv", "final"))
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--out", default="/content/work/results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--bench-offset", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pairs, xi, xs = load(args.work)
    if args.mode == "final":
        model = train_prior(pairs, xi, xs, args)
        state = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
        np.savez(f"{args.out}/prior_{args.tag}.npz", **state)
        json.dump(vars(args) | {"ds": xs.shape[1], "di": xi.shape[1]}, open(f"{args.out}/prior_{args.tag}.json", "w"))
        log("saved final prior")
        return
    rng = np.random.default_rng(0)
    benches = np.array(sorted(pairs.benchmark.unique()))
    fold_of = dict(zip(benches, rng.permutation(len(benches)) % args.folds))
    grid = list(itertools.product((0.8, 1.2, 1.8, 2.5, 3.5), (0.5, 0.8, 1.2), (0.3, 0.45, 0.6, 0.75, 0.9)))
    results = {"grid": {}, "folds": []}
    all_groups, all_draws = [], []
    for fold in range(args.folds):
        test_mask = pairs.benchmark.map(fold_of).to_numpy() == fold
        model = train_prior(pairs[~test_mask], xi, xs, args, seed=fold)
        groups = make_groups(pairs[test_mask], rng)
        if not groups:
            continue
        attach_logits(groups, model, xs, xi)
        draws = sample_draws(groups, rng, args.draws)
        base = {k: simulate_baseline(groups, draws, k) for k in ("const", "beta")}
        fold_res = {"fold": fold, "groups": len(groups), "benchmarks": len({g["bench"] for g in groups}), "baseline": base}
        per_budget = {}
        for hp in grid:
            alc, pb = simulate(groups, draws, None, hp)
            results["grid"].setdefault(str(hp), []).append(alc)
            per_budget[str(hp)] = pb
        best = min(results["grid"], key=lambda k: results["grid"][k][-1])
        fold_res["best_in_fold"] = [best, results["grid"][best][-1], per_budget[best]]
        results.setdefault("per_budget", {}).setdefault("all", []).append(per_budget)
        results["folds"].append(fold_res)
        log(f"fold {fold}", json.dumps(fold_res))
        del model
        torch.cuda.empty_cache()
    mean_grid = {k: float(np.mean(v)) for k, v in results["grid"].items()}
    best = min(mean_grid, key=mean_grid.get)
    pbs = [f[best] for f in results["per_budget"]["all"]]
    results["best"] = [best, mean_grid[best], np.mean(pbs, axis=0).tolist()]
    results["baseline_mean"] = {k: float(np.mean([f["baseline"][k][0] for f in results["folds"]])) for k in ("const", "beta")}
    json.dump(results, open(f"{args.out}/cv_{args.tag}.json", "w"), indent=1)
    beta_pb = np.mean([f["baseline"]["beta"][1] for f in results["folds"]], axis=0).tolist()
    log("BEST", best, round(mean_grid[best], 5), "per budget", np.round(results["best"][2], 4).tolist())
    log("BETA", round(results["baseline_mean"]["beta"], 5), "per budget", np.round(beta_pb, 4).tolist())


if __name__ == "__main__":
    main()

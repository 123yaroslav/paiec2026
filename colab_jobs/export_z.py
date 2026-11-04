"""Train fold priors and export held-out logits, plus one final prior on all data.

Output in /content/work/results
  pairs_z.parquet    benchmark, subject_key, item_key, n, k, i (item row), fold, z (held-out logit)
  prior_<tag>.npz    final prior trained on every benchmark
  prior_<tag>.json   its configuration
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/content/run3")
sys.path.insert(0, "/content/run")
import train_eval3 as te  # noqa: E402

ap = argparse.ArgumentParser()
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
ap.add_argument("--bench-offset", type=int, default=1)
ap.add_argument("--tag", default="B")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
pairs, xi, xs = te.load(args.work)
rng = np.random.default_rng(0)
benches = np.array(sorted(pairs.benchmark.unique()))
fold_of = dict(zip(benches, rng.permutation(len(benches)) % args.folds))
pairs["fold"] = pairs.benchmark.map(fold_of).astype(int)
pairs["z"] = np.nan
for fold in range(args.folds):
    test = (pairs.fold == fold).to_numpy()
    model = te.train_prior(pairs[~test], xi, xs, args, seed=fold)
    sub = pairs[test]
    pairs.loc[test, "z"] = te.base_logits(model, xs, xi, sub.s.to_numpy(), sub.i.to_numpy())
    te.log("fold", fold, "held-out pairs", int(test.sum()))
    del model
    torch.cuda.empty_cache()
pairs[["benchmark", "subject_key", "item_key", "n", "k", "i", "fold", "z"]].to_parquet(f"{args.out}/pairs_z.parquet")
model = te.train_prior(pairs, xi, xs, args, seed=99)
state = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
np.savez(f"{args.out}/prior_{args.tag}.npz", **state)
json.dump(vars(args) | {"ds": xs.shape[1], "di": xi.shape[1]}, open(f"{args.out}/prior_{args.tag}.json", "w"))
te.log("exported held-out logits and final prior", args.tag)

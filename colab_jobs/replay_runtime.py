"""Replay a real submission runtime on the simulator's draws.

The draws are rebuilt with the same seeds, filters and code as sim_official.py,
so per-group ALC can be paired with the simulator's dumped rows. The runtime
gets the same [subject, item] inputs and the full labeled list of the draw, as
on the platform.

usage: replay_runtime.py UNPACKED_SUBMISSION_DIR OUT_PARQUET --seeds 0:30 [--min-date D] [--exclude a,b]
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")))
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_official as so  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("submission")
ap.add_argument("out")
ap.add_argument("--seeds", default="0:20")
ap.add_argument("--min-date", default="")
ap.add_argument("--exclude", default="")
ap.add_argument("--pairs-z", default=PAIEC_ROOT + "/colab_jobs/results/exportB/work/results/pairs_z.parquet")
args = ap.parse_args()

C = PAIEC_ROOT + "/data/corpus_full"
pairs = pd.read_parquet(args.pairs_z)
subjects = pd.read_parquet(f"{C}/subjects.parquet")
dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
if args.min_date:
    pairs = pairs[pairs.subject_key.isin(set(subjects.subject_key[dates >= pd.Timestamp(args.min_date)]))]
if args.exclude:
    pairs = pairs[~pairs.benchmark.isin(set(args.exclude.split(",")))]
so.G.update({"index": so.build_index(pairs)})
items = pd.read_parquet(f"{C}/items.parquet", columns=["item_key", "content", "item_features", "interactors"]).set_index("item_key")
subj = {r["subject_key"]: {k: str(r[k] or "") for k in so_fields} for so_fields in [("normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort", "harness_version", "subject_features_extra")] for r in subjects.to_dict("records")}

sys.path.insert(0, os.path.abspath(args.submission))
import model as runtime  # noqa: E402
for mod in ("prior_runtime",):
    if mod in sys.modules:
        sys.modules[mod].EMBEDDING_CACHE_LIMIT = 10 ** 6
        sys.modules[mod].BASE_LOGIT_CACHE_LIMIT = 10 ** 6
if hasattr(runtime, "CACHE_LIMIT"):
    runtime.CACHE_LIMIT = 10 ** 6


def alias(bench):
    return "benchmark_" + str(int(hashlib.sha256(bench.encode()).hexdigest()[:8], 16) % 900000 + 100000)


def pair(skey, ikey, bench):
    r = items.loc[ikey]
    return [subj[skey], {"item_content": r.content or "", "item_features": r.item_features or "",
                         "interactors": r.interactors or "", "benchmark_id": alias(bench)}]


rows = []
lo, hi = (int(x) for x in args.seeds.split(":"))
t0 = time.time()
for seed in range(lo, hi):
    draw = so.make_draw(np.random.default_rng(seed))
    for budget in so.BUDGETS:
        labeled = []
        for h in draw:
            m = min(budget, len(h["acq"]))
            for ikey, y in zip(h["acq"].index[:m], h["y"][:m]):
                labeled.append([pair(h["s"], ikey, h["bench"]), int(y)])
        for gi, g in enumerate(draw):
            p = np.array([runtime.predict(pair(g["s"], ikey, g["bench"]), labeled) for ikey in g["eval"].index])
            rows.append({"seed": seed, "group": gi, "budget": budget, "brier": so.brier(p, g["eval"])})
    print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    alc = df.pivot_table(index=["seed", "group"], columns="budget", values="brier")
    alc["alc"] = alc[list(so.BUDGETS)].to_numpy() @ so.WEIGHTS
    alc.reset_index().to_parquet(args.out)

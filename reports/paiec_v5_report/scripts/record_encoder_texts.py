"""Record which texts the L5 runtime sends to its encoder at every budget.

The platform restarts the prediction worker at every budget, so in-memory caches
last one budget. This script clears them at every budget and replaces the encoder
with a stub that returns random unit vectors, so it runs in seconds. The output
contains item texts from measurement-db and stays local. time_encoder.py then
times the real encoder on these texts.

usage: python -B scripts/record_encoder_texts.py UNZIPPED_L5_DIR OUT_JSON --seed0 300000 --draws 3
"""


import os as _os
PAIEC_ROOT = _os.environ.get("PAIEC_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = PAIEC_ROOT + ""
sys.path.insert(0, f"{ROOT}/colab_jobs")
import sim_official as so  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("submission")
ap.add_argument("out")
ap.add_argument("--seed0", type=int, default=300000)
ap.add_argument("--draws", type=int, default=3)
ap.add_argument("--min-date", default="2025-01-01")
args = ap.parse_args()

C = f"{ROOT}/data/corpus_full"
pairs = pd.read_parquet(f"{ROOT}/colab_jobs/results/exportB/work/results/pairs_z.parquet")
subjects = pd.read_parquet(f"{C}/subjects.parquet")
dates = pd.to_datetime(subjects.release_date.str.slice(0, 10), errors="coerce")
pairs = pairs[pairs.subject_key.isin(set(subjects.subject_key[dates >= pd.Timestamp(args.min_date)]))]
so.G.update({"index": so.build_index(pairs)})
items = pd.read_parquet(f"{C}/items.parquet", columns=["item_key", "content", "item_features", "interactors"]).set_index("item_key")
fields = ("normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort",
          "harness_version", "subject_features_extra")
subj = {r["subject_key"]: {k: str(r[k] or "") for k in fields} for r in subjects.to_dict("records")}
sys.path.insert(0, os.path.abspath(args.submission))
import model as runtime  # noqa: E402


def alias(bench):
    return "benchmark_" + str(int(hashlib.sha256(bench.encode()).hexdigest()[:8], 16) % 900000 + 100000)


def pair(skey, ikey, bench):
    r = items.loc[ikey]
    return [subj[skey], {"item_content": r.content or "", "item_features": r.item_features or "",
                         "interactors": r.interactors or "", "benchmark_id": alias(bench)}]


class Stub:
    def __init__(self):
        self.log, self.rng = [], np.random.default_rng(0)

    def encode(self, texts, batch_size=1):
        self.log.extend(texts)
        v = self.rng.normal(size=(len(texts), 384)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True), {}


stub = Stub()
runtime._encoder = lambda: stub
uniq, index, runs = [], {}, []
for seed in range(args.seed0, args.seed0 + args.draws):
    draw = so.make_draw(np.random.default_rng(seed))
    for b in so.BUDGETS:
        labeled = []
        for h in draw:
            m = min(b, len(h["acq"]))
            labeled += [[pair(h["s"], i, h["bench"]), int(y)] for i, y in zip(h["acq"].index[:m], h["y"][:m])]
        for name in ("_ITEM_CACHE", "_SUBJECT_CACHE", "_LOGIT_CACHE", "_POSTERIOR_CACHE"):
            getattr(runtime, name).clear()
        stub.log = []
        calls = 0
        for g in draw:
            for i in g["eval"].index:
                runtime.predict(pair(g["s"], i, g["bench"]), labeled)
                calls += 1
        ids = []
        for t in stub.log:
            if t not in index:
                index[t] = len(uniq)
                uniq.append(t)
            ids.append(index[t])
        runs.append({"seed": seed, "budget": b, "groups": len(draw), "calls": calls, "texts": ids})
        print(seed, b, "calls", calls, "encoded", len(ids), flush=True)
json.dump({"texts": uniq, "runs": runs}, open(args.out, "w"))
print("unique texts", len(uniq))

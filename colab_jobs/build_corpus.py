"""Build the full measurement-db training corpus on a Colab VM.

Output in /content/work
  benchmarks.parquet   one row per kept binary benchmark
  subjects.parquet     subject_key and the eight runtime fields
  items.parquet        item_key, benchmark, runtime item fields, text length
  pairs.parquet        benchmark, subject_key, item_key, n trials, k successes
  item_emb.npy         float16 E5 embeddings in items.parquet order
  stats.json           counts per benchmark
"""

import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "aims-foundations/measurement-db"
REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
RAW = os.environ.get("RAW", "/content/mdb")
OUT = os.environ.get("OUT", "/content/work")
RUNTIME_DIR = os.environ.get("RUNTIME_DIR", "/content/run")
EMBED = os.environ.get("EMBED", "1") == "1"
MAX_ITEMS_PER_BENCH = int(os.environ.get("MAX_ITEMS_PER_BENCH", "4000"))
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
sys.path.insert(0, RUNTIME_DIR)
import spec as runtime_spec  # noqa: E402
import textproj  # noqa: E402


def log(*parts):
    print(f"[{time.time() - T0:7.0f}s]", *parts, flush=True)


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value)
    return "" if text in ("None", "nan", "NaN", "null") else text


def key(*parts):
    return hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).hexdigest()


def fetch(patterns):
    for attempt in range(5):
        try:
            return snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=patterns,
                                     local_dir=RAW, max_workers=int(os.environ.get("HF_WORKERS", "16")))
        except Exception as exc:
            log("download retry", attempt, repr(exc)[:200])
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("download failed")


# 1. Benchmark metadata, keep binary benchmarks only.
SKIP_FETCH = os.environ.get("SKIP_FETCH") == "1"
if not SKIP_FETCH:
    fetch(["*/formatted_tables/benchmarks.parquet", "*/formatted_tables/subjects.parquet"])
rows = []
for bench in sorted(os.listdir(RAW)):
    path = f"{RAW}/{bench}/formatted_tables/benchmarks.parquet"
    if os.path.exists(path):
        meta = pq.read_table(path).to_pylist()[0]
        rows.append({k: clean(v) for k, v in meta.items()} | {"dir": bench})
bench_meta = pd.DataFrame(rows)
binary = bench_meta[(bench_meta.response_type == "binary") | (bench_meta.n_response_values == "2")]
log("benchmarks", len(bench_meta), "binary", len(binary))
if not SKIP_FETCH:
    fetch([f"{b}/formatted_tables/responses.parquet" for b in binary.dir] +
          [f"{b}/formatted_tables/items.parquet" for b in binary.dir])
log("downloaded responses and items")

# 2. Per benchmark aggregation.
subjects_all, items_all, pairs_all, stats = {}, [], [], {}
rng = np.random.default_rng(0)
for bench in binary.dir:
    base = f"{RAW}/{bench}/formatted_tables"
    try:
        subs = pq.read_table(f"{base}/subjects.parquet").to_pandas()
        resp = pq.read_table(f"{base}/responses.parquet",
                             columns=["subject_id", "item_id", "response", "interactors"]).to_pandas()
    except Exception as exc:
        log("skip", bench, repr(exc)[:150])
        continue
    resp["y"] = pd.to_numeric(resp["response"], errors="coerce")
    resp = resp[resp.y.isin([0.0, 1.0])]
    if resp.empty:
        stats[bench] = {"skipped": "no binary rows"}
        continue
    sub_key = {}
    for row in subs.itertuples(index=False):
        fields = {f: clean(getattr(row, f, "")) for f in SUBJECT_FIELDS}
        skey = key(*fields.values())
        sub_key[row.subject_id] = skey
        subjects_all.setdefault(skey, fields)
    resp["subject_key"] = resp.subject_id.map(sub_key)
    resp = resp.dropna(subset=["subject_key"])
    resp["interactors"] = resp["interactors"].map(clean)
    agg = resp.groupby(["subject_key", "item_id"], sort=False).agg(
        n=("y", "size"), k=("y", "sum"), interactors=("interactors", "first")).reset_index()
    item_ids = agg.item_id.unique()
    if len(item_ids) > MAX_ITEMS_PER_BENCH:
        coverage = agg.groupby("item_id").size()
        # Keep the best covered items, random among ties, so every subject keeps long groups.
        order = coverage.sample(frac=1, random_state=0).sort_values(ascending=False, kind="stable")
        keep = set(order.index[:MAX_ITEMS_PER_BENCH])
        agg = agg[agg.item_id.isin(keep)]
        item_ids = np.array(sorted(keep))
    items = pq.read_table(f"{base}/items.parquet", columns=["item_id", "content", "item_features"]).to_pandas()
    items = items[items.item_id.isin(set(item_ids))]
    inter = agg.groupby("item_id").interactors.agg(lambda s: s.mode().iat[0] if len(s) else "")
    items["interactors"] = items.item_id.map(inter).fillna("")
    items["content"] = items.content.map(clean)
    items["item_features"] = items.item_features.map(clean)
    items["item_key"] = [key(bench, i) for i in items.item_id]
    items["benchmark"] = bench
    items["text_chars"] = items.content.str.len()
    # Same projection as the runtime: chat JSON -> system + final user task, bounded characters.
    items["text"] = [textproj.item_text(c, f) for c, f in zip(items.content, items.item_features)]
    id2key = dict(zip(items.item_id, items.item_key))
    agg["item_key"] = agg.item_id.map(id2key)
    agg = agg.dropna(subset=["item_key"])
    agg["benchmark"] = bench
    group_sizes = agg.groupby("subject_key").size()
    stats[bench] = {"subjects": int(agg.subject_key.nunique()), "items": int(len(items)),
                    "pairs": int(len(agg)), "trials": int(agg.n.sum()), "mean_y": float(agg.k.sum() / agg.n.sum()),
                    "groups_80": int((group_sizes >= 80).sum())}
    items_all.append(items[["item_key", "benchmark", "content", "item_features", "interactors", "text_chars", "text"]])
    pairs_all.append(agg[["benchmark", "subject_key", "item_key", "n", "k"]])
    log(bench, stats[bench])
    del resp, agg, items
    gc.collect()

items_df = pd.concat(items_all, ignore_index=True).drop_duplicates("item_key")
pairs_df = pd.concat(pairs_all, ignore_index=True)
subjects_df = pd.DataFrame([{"subject_key": k, **v} for k, v in subjects_all.items()])
binary.to_parquet(f"{OUT}/benchmarks.parquet")
subjects_df.to_parquet(f"{OUT}/subjects.parquet")
pairs_df.to_parquet(f"{OUT}/pairs.parquet")
items_df.to_parquet(f"{OUT}/items.parquet")
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
items_df[["item_key", "text"]].to_parquet(f"{OUT}/texts.parquet", compression="zstd")
subjects_df.assign(text=[textproj.subject_text(r) for r in subjects_df.to_dict("records")])[["subject_key", "text"]].to_parquet(
    f"{OUT}/subject_texts.parquet", compression="zstd")
log("corpus", "items", len(items_df), "pairs", len(pairs_df), "subjects", len(subjects_df))

if not EMBED:
    log("EMBED=0, stopping before embeddings")
    sys.exit(0)

# 3. Embeddings with the exact runtime recipe.
import torch
from transformers import AutoModel, AutoTokenizer
import encoder as runtime_encoder

tok = AutoTokenizer.from_pretrained(runtime_spec.MODEL_ID, revision=runtime_spec.REVISION, use_fast=True)
model = AutoModel.from_pretrained(runtime_spec.MODEL_ID, revision=runtime_spec.REVISION,
                                  torch_dtype=torch.float16).cuda().eval()
texts = items_df.text.tolist()
order = np.argsort([len(t) for t in texts])
emb = np.zeros((len(texts), 384), dtype=np.float16)
batch = 128
with torch.inference_mode():
    for start in range(0, len(order), batch):
        idx = order[start:start + batch]
        prepared = runtime_encoder.prepare_texts([texts[i] for i in idx])
        enc = runtime_encoder.tokenize_coverage(tok, prepared)
        enc = {k: v.cuda() for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state.float()
        pooled = runtime_encoder.masked_normalized_pool(hidden, enc["attention_mask"])
        emb[idx] = pooled.cpu().numpy().astype(np.float16)
        if (start // batch) % 200 == 0:
            log("embedded", start + len(idx), "of", len(order))
np.save(f"{OUT}/item_emb.npy", emb)
log("done embeddings", emb.shape)

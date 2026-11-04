"""Zero-shot difficulty ratings from an instruct LLM on a fixed item sample.

The rating is the expected digit 1..9 under the next-token distribution after
the prompt, one forward pass per item. Output /content/work/judge/<tag>.parquet
with item_key, rating and the nine digit probabilities.

env: JUDGE_MODEL, JUDGE_REVISION, PER_BENCH (items per benchmark), TAG, MAX_TOKENS
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content/run")
import judge_prompt  # noqa: E402

T0 = time.time()
MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-8B")
REV = os.environ.get("JUDGE_REVISION") or None
PER_BENCH = int(os.environ.get("PER_BENCH", "120"))
TAG = os.environ.get("TAG", "qwen3_8b")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
OUT = "/content/work/judge"
os.makedirs(OUT, exist_ok=True)


def log(*parts):
    print(f"[{time.time() - T0:6.0f}s]", *parts, flush=True)


items = pd.read_parquet("/content/work/items.parquet", columns=["item_key", "benchmark"]).merge(
    pd.read_parquet("/content/work/texts.parquet"), on="item_key")
if PER_BENCH > 0:
    items = items.groupby("benchmark", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), PER_BENCH), random_state=1)).reset_index(drop=True)
log("items to rate", len(items), "benchmarks", items.benchmark.nunique())

tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, torch_dtype=torch.bfloat16).cuda().eval()
digits = [tok.encode(str(d), add_special_tokens=False)[0] for d in range(1, 10)]
log("model loaded", MODEL, "digit ids", digits)

prompts = [judge_prompt.build(tok, t, MAX_TOKENS) for t in items.text]
lengths = np.array([len(tok.encode(p, add_special_tokens=False)) for p in prompts])
order = np.argsort(lengths, kind="stable")
ckpt = f"{OUT}/{TAG}.partial.npz"
probs = np.full((len(prompts), 9), np.nan, dtype=np.float32)
if os.path.exists(ckpt):
    saved = np.load(ckpt, allow_pickle=True)
    if list(saved["keys"]) == list(items.item_key):
        probs = saved["probs"]
        log("resumed", int((~np.isnan(probs[:, 0])).sum()), "rated items")
TOKENS_PER_BATCH = int(os.environ.get("TOKENS_PER_BATCH", "12000"))
todo = [i for i in order if np.isnan(probs[i, 0])]
batches, cur, cur_max = [], [], 0
for i in todo:
    m = max(cur_max, lengths[i])
    if cur and m * (len(cur) + 1) > TOKENS_PER_BATCH:
        batches.append(cur)
        cur, m = [], lengths[i]
    cur.append(i)
    cur_max = m
if cur:
    batches.append(cur)
done = len(prompts) - len(todo)
with torch.inference_mode():
    for bi, idx in enumerate(batches):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        # Only the last position is needed. Full-sequence logits do not fit for long prompts.
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        probs[idx] = torch.softmax(logits[:, digits], dim=-1).cpu().numpy()
        done += len(idx)
        if bi % 50 == 0:
            np.savez(ckpt, keys=np.array(items.item_key), probs=probs)
            log("rated", done, "of", len(prompts), "max tokens", int(lengths[idx].max()))
rating = probs @ np.arange(1, 10, dtype=np.float32)
out = pd.DataFrame({"item_key": items.item_key, "benchmark": items.benchmark, "rating": rating})
for d in range(9):
    out[f"p{d + 1}"] = probs[:, d]
out.to_parquet(f"{OUT}/{TAG}.parquet")
log("saved", len(out), "mean rating", float(rating.mean()), "std", float(rating.std()))

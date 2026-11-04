"""Item features from two Qwen models on a fixed subset, for comparison with E5.

Subset is up to 600 items per benchmark, seed 0. Output in /content/work/bakeoff
  keys.parquet       item_key order shared by all feature files
  qwen3emb06.npy     Qwen3-Embedding-0.6B last-token embedding, normalized
  qwen3_4b_mid.npy   Qwen3-4B mean of layer 24 hidden states, normalized
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

T0 = time.time()
OUT = "/content/work/bakeoff"
os.makedirs(OUT, exist_ok=True)
MAX_TOKENS = 512


def log(*parts):
    print(f"[{time.time() - T0:6.0f}s]", *parts, flush=True)


items = pd.read_parquet("/content/work/items.parquet", columns=["item_key", "benchmark"])
texts = pd.read_parquet("/content/work/texts.parquet")
items = items.merge(texts, on="item_key")
subset = items.groupby("benchmark", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 600), random_state=0))
subset = subset.reset_index(drop=True)
subset[["item_key", "benchmark"]].to_parquet(f"{OUT}/keys.parquet")
log("subset", len(subset), "items from", subset.benchmark.nunique(), "benchmarks")
order = np.argsort(subset.text.str.len().to_numpy(), kind="stable")
texts = subset.text.tolist()


def run(name, tok, forward, dim, batch):
    out = np.zeros((len(texts), dim), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt").to("cuda")
            vec = torch.nn.functional.normalize(forward(enc).float(), dim=-1)
            out[idx] = vec.cpu().numpy().astype(np.float16)
            if (start // batch) % 50 == 0:
                log(name, start + len(idx), "of", len(order))
    np.save(f"{OUT}/{name}.npy", out)
    log("saved", name, out.shape)


# Qwen3-Embedding-0.6B. Last-token pooling with left padding, as in its model card.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                                    padding_side="left")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                                  torch_dtype=torch.float16).cuda().eval()
run("qwen3emb06", tok, lambda enc: model(**enc).last_hidden_state[:, -1], model.config.hidden_size, 64)
del model
torch.cuda.empty_cache()

# Qwen3-4B base hidden states. Mean over tokens of a middle layer.
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c", padding_side="right")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
                                             torch_dtype=torch.float16).cuda().eval()
LAYER = 24


def mid_mean(enc):
    hidden = model(**enc, output_hidden_states=True).hidden_states[LAYER]
    mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1)


run("qwen3_4b_mid", tok, mid_mean, model.config.hidden_size, 16)
log("done")

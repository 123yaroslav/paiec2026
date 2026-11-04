"""Embed item and subject texts on a Colab GPU with the runtime E5 recipe."""

import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/content/run")
import encoder  # noqa: E402
import spec  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = True
tok = AutoTokenizer.from_pretrained(spec.MODEL_ID, revision=spec.REVISION, use_fast=True)
model = AutoModel.from_pretrained(spec.MODEL_ID, revision=spec.REVISION, torch_dtype=torch.float32).cuda().eval()
print("model loaded", f"{time.time() - T0:.0f}s", torch.cuda.get_device_name(0), flush=True)


def embed(texts, name, batch=192):
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), 384), dtype=np.float16)
    with torch.inference_mode():
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            enc = encoder.tokenize_coverage(tok, encoder.prepare_texts([texts[i] for i in idx]))
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            out[idx] = encoder.masked_normalized_pool(hidden, enc["attention_mask"]).cpu().numpy().astype(np.float16)
            if (start // batch) % 100 == 0:
                print(f"[{time.time() - T0:6.0f}s] {name} {start + len(idx)} of {len(order)}", flush=True)
    return out


subjects = pd.read_parquet("/content/work/subject_texts.parquet")
np.save("/content/work/subject_emb.npy", embed(subjects.text.tolist(), "subjects"))
items = pd.read_parquet("/content/work/texts.parquet")
np.save("/content/work/item_emb.npy", embed(items.text.tolist(), "items"))
print(f"[{time.time() - T0:6.0f}s] done items {len(items)} subjects {len(subjects)}", flush=True)

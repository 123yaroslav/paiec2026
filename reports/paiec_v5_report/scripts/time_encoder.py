"""Time L5's encoder call by call on a clean CPU machine.

Encodes every recorded text once with batch size 1, as the runtime does, and
sums the latencies of the texts that each cold budget encodes.
usage: python time_encoder.py TEXTS_JSON OUT_JSON  (run next to v5/encoder.py on a CPU machine)
"""
import hashlib, json, os, platform, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from huggingface_hub import hf_hub_download
import encoder

REPO, REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SHA = {
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
}
src, out = sys.argv[1], sys.argv[2]
d = os.path.join(os.path.dirname(os.path.abspath(out)), "e5")
os.makedirs(d, exist_ok=True)
for name, sha in SHA.items():
    p = hf_hub_download(REPO, name, revision=REV, local_dir=d, token=False)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha, name
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
info = {"cpu": cpu, "nproc": os.cpu_count(), "torch": torch.__version__, "threads": torch.get_num_threads()}
print(json.dumps(info), flush=True)
loads = []
for _ in range(3):
    t = time.perf_counter(); enc = encoder.FrozenEncoder(d, device="cpu"); loads.append(time.perf_counter() - t)
data = json.load(open(src))
texts = data["texts"]
for t_ in texts[:5]:
    enc.encode([t_], batch_size=1)
lat = []
t0 = time.perf_counter()
for i, text in enumerate(texts):
    t = time.perf_counter(); enc.encode([text], batch_size=1); lat.append(time.perf_counter() - t)
    if i % 100 == 0:
        print(f"{i}/{len(texts)} mean {statistics.fmean(lat):.3f}s elapsed {time.perf_counter() - t0:.0f}s", flush=True)
by_seed = {}
for r in data["runs"]:
    by_seed.setdefault(r["seed"], []).append(sum(lat[j] for j in r["texts"]))
res = {"info": info, "load_s": loads, "per_text_mean": statistics.fmean(lat), "per_text_median": statistics.median(lat),
       "per_text_p90": sorted(lat)[int(0.9 * len(lat))], "n_texts": len(lat),
       "draw_cold_encode_s": {str(k): sum(v) for k, v in by_seed.items()},
       "groups": {str(r["seed"]): r["groups"] for r in data["runs"]},
       "latencies": lat}
json.dump(res, open(out, "w"))
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "latencies"}), flush=True)

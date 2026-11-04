"""PAIEC V5 predictor.

Base logit from a two-tower network over frozen E5 embeddings of the item text
and of the subject description, trained on the binary benchmarks of
measurement-db. Labels on the same benchmark update a shared benchmark offset
and a per-subject offset through adapt.posterior.
"""

import hashlib
import json
import math
import threading
from pathlib import Path

import numpy as np

try:
    from . import adapt, encoder, textproj
except ImportError:
    import adapt
    import encoder
    import textproj

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "v5_config.json").read_text())
_LOCK = threading.RLock()
_STATE = {}
_ITEM_CACHE = {}
_SUBJECT_CACHE = {}
_LOGIT_CACHE = {}
CACHE_LIMIT = 20000


def _encoder():
    if "encoder" not in _STATE:
        _STATE["encoder"] = encoder.FrozenEncoder(str(ROOT / "encoder_assets"), device="cpu")
    return _STATE["encoder"]


def _prior():
    if "prior" not in _STATE:
        import torch
        import torch.nn as nn

        cfg = CONFIG["prior"]
        hidden, rank = cfg["hidden"], cfg["rank"]

        class Prior(nn.Module):
            def __init__(self):
                super().__init__()
                self.s = nn.Sequential(nn.Linear(cfg["ds"], hidden), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden, hidden), nn.GELU())
                self.i = nn.Sequential(nn.Linear(cfg["di"], hidden), nn.GELU(), nn.Dropout(0.0), nn.Linear(hidden, hidden), nn.GELU())
                self.a = nn.Linear(hidden, 1)
                self.b = nn.Linear(hidden, 1)
                self.u = nn.Linear(hidden, rank, bias=False)
                self.v = nn.Linear(hidden, rank, bias=False)

            def forward(self, xs, xi):
                hs, hi = self.s(xs), self.i(xi)
                inter = (self.u(hs) * self.v(hi)).sum(-1) / math.sqrt(rank)
                return self.a(hs).squeeze(-1) + self.b(hi).squeeze(-1) + inter

        model = Prior()
        weights = np.load(ROOT / "prior.npz")
        model.load_state_dict({k: torch.tensor(weights[k]) for k in weights.files})
        model.eval()
        _STATE["prior"] = model
    return _STATE["prior"]


def _embed(text, cache):
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vector = cache.get(key)
    if vector is None:
        array, _ = _encoder().encode([text], batch_size=1)
        vector = array[0].astype(np.float32)
        if len(cache) >= CACHE_LIMIT:
            cache.clear()
        cache[key] = vector
    return vector


def _years(date):
    import pandas as pd
    parsed = pd.to_datetime(pd.Series([str(date or "")[:10]]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return 0.0, 1.0
    return float(min(5.0, max(-5.0, (parsed - pd.Timestamp("2023-01-01")).days / 365.25))), 0.0


def item_key(item):
    content = item.get("item_content", "") or ""
    features = item.get("item_features", "") or ""
    return hashlib.sha256((features + "\x1f" + content).encode("utf-8")).hexdigest()


def item_vector(item):
    content = item.get("item_content", "") or ""
    features = item.get("item_features", "") or ""
    with _LOCK:
        return _embed(textproj.item_text(content, features), _ITEM_CACHE)


def date_offset(subject):
    """Logit offset from the release date, fitted on training groups (colab_jobs/results/date_prior.json)."""
    dp = CONFIG["adapt"].get("date_prior")
    if not dp:
        return 0.0
    years, missing = _years(subject.get("release_date", ""))
    if missing:
        return dp["a0"]
    return dp["a"] + dp["b"] * min(dp["clip"][1], max(dp["clip"][0], years))


NAME_ABILITY = json.loads((ROOT / "name_ability.json").read_text()) if (ROOT / "name_ability.json").exists() else {}


def name_offset(subject):
    """Average within-benchmark skill of this model name on training benchmarks, shrunk and clipped."""
    scale = CONFIG["adapt"].get("name_scale", 0.0)
    if not scale:
        return 0.0
    name = " ".join(str(subject.get("normalized_name", "") or "").lower().split())
    return scale * max(-2.5, min(2.5, NAME_ABILITY.get(name, 0.0)))


def base_logit(subject, item):
    """Prior logit in units of the adaptation model: gamma * base_logit is the item term."""
    return _prior_logit(subject, item) + (date_offset(subject) + name_offset(subject)) / CONFIG["adapt"]["gamma"]


def _prior_logit(subject, item):
    if not CONFIG["adapt"].get("use_prior", True):
        return 0.0
    content = item.get("item_content", "") or ""
    features = item.get("item_features", "") or ""
    key = (textproj.subject_key(subject), item_key(item))
    with _LOCK:
        if key in _LOGIT_CACHE:
            return _LOGIT_CACHE[key]
        import torch
        subj_vec = _embed(textproj.subject_text(subject), _SUBJECT_CACHE)
        item_vec = _embed(textproj.item_text(content, features), _ITEM_CACHE)
        xs = np.concatenate([subj_vec, np.array(_years(subject.get("release_date", "")), dtype=np.float32)])
        xi = np.concatenate([item_vec, np.array([math.log1p(len(content)) / 10], dtype=np.float32)])
        with torch.no_grad():
            value = float(_prior()(torch.tensor(xs[None]), torch.tensor(xi[None]))[0])
        value = max(-15.0, min(15.0, value)) if math.isfinite(value) else 0.0
        if len(_LOGIT_CACHE) >= CACHE_LIMIT:
            _LOGIT_CACHE.clear()
        _LOGIT_CACHE[key] = value
        return value


_POSTERIOR_CACHE = {}


def _benchmark_state(bench, labeled, hp):
    """Labels of one benchmark with logits, subject indices and level probabilities, cached per label set."""
    entries = [(e[0], float(e[1])) for e in labeled or [] if e[0][1].get("benchmark_id", "") == bench]
    sig = hashlib.sha256(json.dumps([[textproj.subject_key(p[0]), item_key(p[1]), y] for p, y in entries]).encode()).hexdigest()
    with _LOCK:
        if (bench, sig) in _POSTERIOR_CACHE:
            return _POSTERIOR_CACHE[(bench, sig)]
    index, zs, ss, ys, keys = {}, [], [], [], []
    for pair, y in entries:
        idx = index.setdefault(textproj.subject_key(pair[0]), len(index))
        zs.append(base_logit(pair[0], pair[1]))
        ss.append(idx)
        ys.append(y)
        keys.append(item_key(pair[1]))
    z, s, y = np.array(zs), np.array(ss, dtype=int), np.array(ys)
    lab_p = np.empty(len(y))
    for sj in set(ss):
        mj, _ = adapt.posterior(z, s, y, sj, hp["tau_c"], hp["tau_d"], hp["gamma"])
        msk = s == sj
        lab_p[msk] = 1 / (1 + np.exp(-(hp["gamma"] * z[msk] + mj)))
    by_item = {}
    for j, k in enumerate(keys):
        by_item.setdefault(k, []).append(j)
    x_lab = None
    if hp.get("item_kernel") and entries:
        x_lab = np.stack([item_vector(pair[1]) for pair, _ in entries])
    state = {"index": index, "z": z, "s": s, "y": y, "resid": y - lab_p, "w": lab_p * (1 - lab_p),
             "by_item": by_item, "x_lab": x_lab}
    with _LOCK:
        if len(_POSTERIOR_CACHE) > 256:
            _POSTERIOR_CACHE.clear()
        _POSTERIOR_CACHE[(bench, sig)] = state
    return state


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    hp = CONFIG["adapt"]
    bench = item.get("benchmark_id", "")
    z_target = base_logit(subject, item)
    st = _benchmark_state(bench, labeled, hp)
    index = dict(st["index"])
    target = index.setdefault(textproj.subject_key(subject), len(index))
    mean, var = adapt.posterior(st["z"], st["s"], st["y"], target, hp["tau_c"], hp["tau_d"], hp["gamma"])
    eta_shift = 0.0
    sig2 = hp.get("item_exact", 0.0)
    js = st["by_item"].get(item_key(item))
    if sig2 and js:
        # Same item already labeled on this benchmark, for this or another subject.
        eta_shift = sig2 * st["resid"][js].sum() / (1 + sig2 * st["w"][js].sum())
    if hp.get("item_kernel") and st["x_lab"] is not None:
        bandwidth, kernel_sig2 = hp["item_kernel"]
        eta_shift += float(adapt.item_shift(item_vector(item)[None], st["x_lab"], st["resid"], st["w"],
                                            bandwidth, kernel_sig2)[0])
    p = float(adapt.predict(np.array([z_target + eta_shift / hp["gamma"]]), mean, var, hp["gamma"])[0])
    if not math.isfinite(p):
        p = 0.5
    return min(max(p, 1e-4), 1 - 1e-4)

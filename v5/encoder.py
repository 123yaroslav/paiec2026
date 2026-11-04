"""Frozen E5 pooling with bounded generic head, middle and tail token coverage."""

from __future__ import annotations

import time


def prepare_texts(texts, *, max_chars=8192, prefix="query: "):
    if type(max_chars) is not int or not 64 <= max_chars <= 32768:
        raise ValueError("max_chars must be between 64 and 32768")
    if prefix != "query: ":
        raise ValueError("This non-retrieval pilot requires the official query prefix")
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("Texts must be a list of strings")
    return [prefix + text[:max_chars] for text in texts]


def choose_device(requested="auto", *, allow_cpu_fallback=False):
    import torch
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Device must be auto, cpu or mps")
    available = torch.backends.mps.is_available()
    if requested == "auto":
        return "mps" if available else "cpu"
    if requested == "mps" and not available:
        if allow_cpu_fallback:
            return "cpu"
        raise RuntimeError("MPS is unavailable in this execution environment")
    return requested


def masked_normalized_pool(hidden_states, attention_mask):
    import torch
    import torch.nn.functional as functional
    if hidden_states.ndim != 3 or attention_mask.ndim != 2 or hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and attention mask shapes differ")
    counts = attention_mask.sum(dim=1)
    if torch.any(counts <= 0):
        raise ValueError("Every encoded input must contain at least one visible token")
    hidden = hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    pooled = hidden.sum(dim=1) / counts[..., None]
    return functional.normalize(pooled, p=2, dim=1)


def select_token_chunks(tokens, max_tokens=512):
    """Keep query-prefix coverage and task variation across a long token span."""
    if max_tokens != 512:
        raise ValueError("The fitted semantic recipe requires the explicit 512-token limit")
    if len(tokens) <= 510:
        return list(tokens)
    n = len(tokens)
    sizes = (128, 127, 127, 128)
    starts = (0, (n - 127) // 3, 2 * (n - 127) // 3, n - 128)
    return [token for start, size in zip(starts, sizes) for token in tokens[start:start + size]]


def tokenize_coverage(tokenizer, prepared, *, max_tokens=512):
    if tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Pinned E5 tokenizer must wrap exactly two special tokens")
    raw = tokenizer(prepared, add_special_tokens=False, padding=False, truncation=False, verbose=False)
    features = []
    for ids in raw["input_ids"]:
        # Native Transformers 5 exposes the verified XLM-R CLS/SEP IDs but no
        # longer provides build_inputs_with_special_tokens on this tokenizer.
        bounded = [tokenizer.cls_token_id, *select_token_chunks(ids, max_tokens), tokenizer.sep_token_id]
        if not 2 <= len(bounded) <= max_tokens:
            raise ValueError("Semantic tokenizer crossed its fixed position bound")
        features.append({"input_ids": bounded, "attention_mask": [1] * len(bounded)})
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def encode_batches(model, tokenizer, texts, *, device="cpu", batch_size=8,
                   max_tokens=512, max_chars=8192, progress=None):
    import numpy as np
    import torch
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be between one and thirty-two")
    if type(max_tokens) is not int or not 8 <= max_tokens <= 512:
        raise ValueError("max_tokens must be between eight and the encoder limit of 512")
    prepared = prepare_texts(texts, max_chars=max_chars)
    model.eval()
    outputs = []
    token_count = max_length = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = tokenize_coverage(tokenizer, prepared[start:start + batch_size], max_tokens=max_tokens)
            batch = {key: value.to(device) for key, value in batch.items()}
            encoded = model(**batch)
            pooled = masked_normalized_pool(encoded.last_hidden_state, batch["attention_mask"])
            outputs.append(pooled.to("cpu", dtype=torch.float32).numpy())
            token_count += int(batch["attention_mask"].sum().item())
            max_length = max(max_length, int(batch["attention_mask"].shape[1]))
            if progress is not None:
                progress(min(start + batch_size, len(prepared)), len(prepared))
    if device == "mps":
        torch.mps.synchronize()
    dimension = int(model.config.hidden_size)
    array = np.concatenate(outputs, axis=0) if outputs else np.empty((0, dimension), dtype=np.float32)
    validate_embeddings(array, len(texts), dimension)
    return array, {"seconds": time.perf_counter() - started, "input_texts": len(texts),
                   "tokens_including_prefix_and_special_tokens": token_count,
                   "maximum_padded_tokens": max_length, "batch_size": batch_size,
                   "max_tokens": max_tokens, "max_chars": max_chars}


def validate_embeddings(array, count, dimension):
    import numpy as np
    if array.shape != (count, dimension) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Embedding export must have finite float32 values and its declared shape")
    if count and not np.allclose(np.linalg.norm(array, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Embedding rows must be L2 normalized")


class FrozenEncoder:
    def __init__(self, directory, *, device="auto", allow_cpu_fallback=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = choose_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        self.model = AutoModel.from_pretrained(directory, local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True,
                                               attn_implementation="eager", torch_dtype=torch.float32)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)
        if int(self.model.config.hidden_size) != 384:
            raise ValueError("Pinned encoder dimension changed")

    def encode(self, texts, **kwargs):
        return encode_batches(self.model, self.tokenizer, texts, device=self.device, **kwargs)

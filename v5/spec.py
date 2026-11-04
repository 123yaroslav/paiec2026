"""Fixed frozen-encoder recipe shared by training and submission."""

import hashlib
import json
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
MAX_CHARS = 8192
MAX_TOKENS = 512
PROJECTION_VERSION = "validated_chat_target_char_head_middle_tail_v2"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def encoder_recipe():
    return {"model_id": MODEL_ID, "revision": REVISION, "dimension": DIMENSION,
            "prefix": "query: ", "pooling": "attention_mask_mean_then_l2_normalize",
            "max_chars": MAX_CHARS, "max_tokens": MAX_TOKENS,
            "dtype": "float32", "trust_remote_code": False,
            "token_coverage": {"content_limit": 510, "chunks": [128, 127, 127, 128],
                               "starts": "0,(N-127)//3,2*(N-127)//3,N-128"},
            "projection_version": PROJECTION_VERSION,
            "projection_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "encoder_code_sha256": hashlib.sha256(Path(__file__).with_name("encoder.py").read_bytes()).hexdigest()}


def bounded_characters(text, budget):
    if len(text) <= budget:
        return text
    third, middle = (budget - 2) // 3, len(text) // 2
    return (text[:third] + "\n" + text[middle - third // 2:middle + third - third // 2]
            + "\n" + text[-(budget - 2 * third - 2):])


def semantic_text(text):
    """Use final user task plus bounded system text for valid chat inputs only.

    This projection never uses source names or outcomes. It intentionally omits
    earlier demonstrations from semantic features; hashed features retain their
    original complete runtime text projection. Ambiguous formats remain raw.
    """
    if not isinstance(text, str):
        raise ValueError("Encoder task text must be a string")
    try:
        messages = json.loads(text)
    except (ValueError, TypeError):
        return bounded_characters(text, MAX_CHARS)
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant", "tool"}
                   or not isinstance(m.get("content"), str) for m in messages)
            or messages[-1]["role"] != "user" or not messages[-1]["content"].strip()):
        return bounded_characters(text, MAX_CHARS)
    system = "\n".join(m["content"] for m in messages[:-1] if m["role"] == "system")[:512]
    prefix = "System instructions\n" + system + "\n\nUser task\n" if system else "User task\n"
    return prefix + bounded_characters(messages[-1]["content"], MAX_CHARS - len(prefix))


def text_key(text):
    if not isinstance(text, str):
        raise ValueError("Encoder task text must be a string")
    return hashlib.sha256(semantic_text(text).encode("utf-8")).hexdigest()

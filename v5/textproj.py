"""Text projections shared by training and the submission runtime."""

# Import at module load: the official validator restores sys.path before predict().
try:
    from . import spec
except ImportError:
    import spec

SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "reasoning_effort", "harness_version", "subject_features_extra")


def subject_text(subject):
    """One short description of the evaluated system for the E5 encoder."""
    get = lambda key: " ".join(str(subject.get(key, "") or "").split())
    lines = [
        "Model: " + get("normalized_name"),
        "Provider: " + get("provider"),
        "Released: " + get("release_date"),
        "Harness: " + (get("harness") + " " + get("harness_version")).strip(),
        "Reasoning effort: " + get("reasoning_effort"),
        "Extra: " + get("subject_features_extra")[:300],
    ]
    return "\n".join(lines)


def subject_key(subject):
    import hashlib
    raw = "\x1f".join(str(subject.get(k, "") or "") for k in SUBJECT_FIELDS)
    return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()


def item_text(content, features):
    """Item features first, then the V3 chat projection of the task text."""
    feats = " ".join(str(features or "").split())[:300]
    body = spec.semantic_text(str(content or ""))
    return ("Features: " + feats + "\n" + body) if feats else body

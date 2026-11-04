"""Assemble a V5 submission ZIP with the frozen E5 encoder bundled inside.

usage: build_zip.py PRIOR_NPZ PRIOR_JSON TAU_C TAU_D GAMMA OUT_ZIP [EXTRA_ADAPT_JSON]
EXTRA_ADAPT_JSON may set use_prior (false gives a pure level model without the
encoder), item_exact, item_kernel, date_prior and name_scale.

The five encoder files come from intfloat/multilingual-e5-small at the pinned
revision. They are taken from the local V3 archive when it exists and from the
Hugging Face hub otherwise. Both sources have the SHA-256 values in E5_SHA256.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3_ZIP = HERE.parent / "dist" / "semantic_v3_bundled.zip"
CODE = ["model.py", "adapt.py", "encoder.py", "spec.py", "textproj.py", "name_ability.json"]
E5_REPO, E5_REVISION = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_SHA256 = {
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
}


def encoder_files():
    """Yield (name, bytes) of the pinned encoder files, verified by SHA-256."""
    if V3_ZIP.exists():
        with zipfile.ZipFile(V3_ZIP) as src:
            blobs = {name: src.read("encoder_assets/" + name) for name in E5_SHA256}
    else:
        from huggingface_hub import hf_hub_download
        blobs = {name: Path(hf_hub_download(E5_REPO, name, revision=E5_REVISION)).read_bytes() for name in E5_SHA256}
    for name, blob in blobs.items():
        if hashlib.sha256(blob).hexdigest() != E5_SHA256[name]:
            raise SystemExit(f"encoder file {name} does not match the pinned SHA-256")
        yield name, blob


def main():
    prior_npz, prior_json, tau_c, tau_d, gamma, out = sys.argv[1:7]
    extra = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
    cfg = json.loads(Path(prior_json).read_text())
    config = {"prior": {k: cfg[k] for k in ("hidden", "rank", "ds", "di")},
              "adapt": {"tau_c": float(tau_c), "tau_d": float(tau_d), "gamma": float(gamma), **extra},
              "prior_source": Path(prior_npz).name}
    out = Path(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        if extra.get("use_prior", True):
            for name, blob in encoder_files():
                dst.writestr(zipfile.ZipInfo("encoder_assets/" + name, (2026, 1, 1, 0, 0, 0)), blob, zipfile.ZIP_DEFLATED)
        for name in CODE:
            dst.write(HERE / name, name)
        dst.write(prior_npz, "prior.npz")
        dst.writestr("v5_config.json", json.dumps(config, indent=1))
        if extra.get("models_txt"):
            dst.writestr("models.txt", "intfloat/multilingual-e5-small\n")
        dst.writestr("prediction_config.json", json.dumps({"concurrency": 1}))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".manifest.json").write_text(json.dumps({"archive": out.name, "sha256": digest, "config": config}, indent=1) + "\n")
    print(out, digest)


if __name__ == "__main__":
    main()

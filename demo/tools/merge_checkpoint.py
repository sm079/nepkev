"""Fold a Kev LoRA checkpoint into its base and write what the browser demo needs.

Writes to --out:
  hf/          the base model (Qwen3_5ForConditionalGeneration layout) with the adapter merged into its language model,
               fp32, ready for the ONNX Runtime GenAI model builder
  head.bin     the pointer head as little-endian float32: q.weight [dp, d], q.bias [dp], k.weight [dp, d], k.bias [dp]
  head.json    shapes, the calibration temperature and the base it belongs to

Run with the project environment and the kev extra (`uv sync --extra kev`):
  uv run python demo/tools/merge_checkpoint.py --run runs/mix-v3-add/runs/selected --out build/merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from kev.checkpoint import Checkpoint, LoadOptions
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="checkpoint directory or Hub id (e.g. sm079/nepkev)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ck = Checkpoint(args.run)
    meta = ck.meta
    if ck.full:
        raise SystemExit("this script folds a LoRA adapter; a full-weight checkpoint needs no merge")
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, merge=True))
    merged = model.lm.state_dict()

    base = Qwen3_5ForConditionalGeneration.from_pretrained(meta.base, revision=meta.base_revision, dtype=torch.float32)
    base.model.language_model.load_state_dict(merged, strict=True)   # same module tree as Kev's backbone
    hf = args.out / "hf"
    if hf.exists():
        shutil.rmtree(hf)
    base.save_pretrained(hf)
    AutoTokenizer.from_pretrained(meta.base, revision=meta.base_revision).save_pretrained(hf)
    try:
        AutoProcessor.from_pretrained(meta.base, revision=meta.base_revision).save_pretrained(hf)
    except Exception as e:  # the builder only needs the config and weights; the processor is a convenience
        print(f"processor not saved: {e}")

    head = {k: v.detach().float().numpy() for k, v in model.head.state_dict().items()}
    dp, d = head["q.weight"].shape
    parts = [head["q.weight"], head["q.bias"], head["k.weight"], head["k.bias"]]
    (args.out / "head.bin").write_bytes(b"".join(np.ascontiguousarray(p, dtype="<f4").tobytes() for p in parts))
    (args.out / "head.json").write_text(json.dumps({
        "hidden_size": int(d), "pointer_dim": int(dp), "layout": ["q.weight", "q.bias", "k.weight", "k.bias"],
        "dtype": "float32", "temperature": float(meta.temperature), "base": meta.base, "base_revision": meta.base_revision,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {hf} and head ({dp}x{d}, T={meta.temperature:.4f})")


if __name__ == "__main__":
    main()

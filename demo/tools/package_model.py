"""Lay out the browser model folder (uploaded to the model repo as `onnx/`) and its manifest.

  python demo/tools/package_model.py --head build/webgpu/merged --out build/webgpu/hub/onnx \
      --variant q4f16=build/webgpu/onnx-int4kqm-webgpu --variant fp16=build/webgpu/onnx-fp16-webgpu

Layout:
  manifest.json                     what the page reads first: head shape and temperature, files, per variant its
                                    graph inputs (so the page can build empty past-state tensors) and download size
  head.bin                          pointer head, float32 (see merge_checkpoint.py)
  tokenizer.json, tokenizer_config.json
  <variant>/model.onnx, <variant>/model.onnx.data

Every graph is built for the WebGPU execution provider; the same graph also runs on ONNX Runtime's CPU kernels (wasm).
Needs `onnx` (not part of the project environment).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import onnx

ELEM = {1: "float32", 10: "float16", 7: "int64"}


def place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)          # same drive: no second copy of a large file
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True, type=Path, help="merge_checkpoint.py output (head.bin, head.json, hf/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--label", action="append", default=[], metavar="NAME=TEXT", help="short name shown in the weights menu")
    ap.add_argument("--describe", action="append", default=[], metavar="NAME=TEXT", help="one line shown in the page")
    ap.add_argument("--recommend", action="append", default=[], metavar="BACKEND=NAME",
                    help="the variant the page picks for a backend (webgpu, wasm)")
    args = ap.parse_args()

    head = json.loads((args.head / "head.json").read_text(encoding="utf-8"))
    notes = dict(d.split("=", 1) for d in args.describe)
    labels = dict(d.split("=", 1) for d in args.label)
    out = args.out
    files = {}
    for name, src in [("head.bin", args.head / "head.bin"), ("tokenizer.json", args.head / "hf" / "tokenizer.json"),
                      ("tokenizer_config.json", args.head / "hf" / "tokenizer_config.json")]:
        place(src, out / name)
        files[name] = src.stat().st_size

    variants = {}
    for spec in args.variant:
        name, src = spec.split("=", 1)
        src = Path(src)
        graph = onnx.load(str(src / "model.onnx"), load_external_data=False).graph
        genai = json.loads((src / "genai_config.json").read_text(encoding="utf-8"))["model"]["decoder"]
        # an empty past for a first pass: batch 1, no past tokens, the KV head width from the builder's config
        dims = {"batch_size": 1, "past_sequence_length": 0, "kv_cache_dim": genai["head_size"]}
        inputs = []
        for i in graph.input:
            shape = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
            entry = {"name": i.name, "dtype": ELEM[i.type.tensor_type.elem_type], "shape": shape}
            if i.name.startswith("past"):
                unknown = [d for d in shape if isinstance(d, str) and d not in dims]
                if unknown:
                    raise SystemExit(f"{src}: {i.name} has dimensions {unknown} this script cannot size")
                entry["empty_shape"] = [dims[d] if isinstance(d, str) else d for d in shape]
            inputs.append(entry)
        outputs = [o.name for o in graph.output]
        if "hidden_states" not in outputs:
            raise SystemExit(f"{src}: no hidden_states output (build with exclude_lm_head=true)")
        vf = []
        for f in ("model.onnx", "model.onnx.data"):
            place(src / f, out / name / f)
            vf.append({"name": f, "bytes": (src / f).stat().st_size})
        variants[name] = {"dir": name, "files": vf, "bytes": sum(f["bytes"] for f in vf), "inputs": inputs,
                          "label": labels.get(name, name), "note": notes.get(name, "")}

    manifest = {"format": 1, "head": {k: head[k] for k in ("hidden_size", "pointer_dim", "temperature", "layout", "dtype")},
                "base": head["base"], "files": files, "variants": variants,
                "recommended": dict(r.split("=", 1) for r in args.recommend)}
    for backend, name in manifest["recommended"].items():
        if name not in variants:
            raise SystemExit(f"--recommend {backend}={name}: no such variant")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, v in variants.items():
        print(f"{name}: {v['bytes'] / 2**20:.0f} MiB, {len(v['inputs'])} inputs")


if __name__ == "__main__":
    main()

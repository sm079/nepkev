"""Quantize only the token-embedding lookup of a builder export to 4 bits (GatherBlockQuantized).

The model builder cannot emit an int8 embedding, so an int8 export keeps its 248k x 1024 table in fp16, nearly half
the download. The table is only read as a lookup (the LM head is excluded), so 4-bit blocks there perturb the input
vectors and nothing else; the int8 matmuls stay as they are.

  python demo/tools/quantize_embedding.py --model build/webgpu/onnx-int8-webgpu --out build/webgpu/onnx-int8e4-webgpu

Needs onnx and onnxruntime (not part of the project environment).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

EMBED = "/model/embed_tokens/Gather"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help="builder output folder (model.onnx + model.onnx.data)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()

    model = onnx.load(str(args.model / "model.onnx"))
    if not any(n.name == EMBED and n.op_type == "Gather" for n in model.graph.node):
        raise SystemExit(f"{args.model}: no plain {EMBED} node (already quantized?)")
    q = MatMulNBitsQuantizer(model, bits=4, block_size=args.block_size, is_symmetric=True, nodes_to_include=[EMBED],
                             op_types_to_quantize=("Gather",), quant_axes=(("Gather", 1),))
    q.process()

    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.model.iterdir():          # configs and tokenizer files travel with the graph
        if f.name not in ("model.onnx", "model.onnx.data"):
            shutil.copy2(f, args.out / f.name)
    out = args.out / "model.onnx"
    for stale in (out, args.out / "model.onnx.data"):
        stale.unlink(missing_ok=True)
    q.model.save_model_to_file(str(out), use_external_data_format=True)
    data = sorted(p.name for p in args.out.glob("model.onnx*"))
    print(f"wrote {data} to {args.out}")


if __name__ == "__main__":
    main()

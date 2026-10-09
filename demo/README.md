# Browser demo

A static page that runs the fine-tuned router ([`sm079/nepkev`](https://huggingface.co/sm079/nepkev)) entirely in the
browser with ONNX Runtime Web: WebGPU when the browser has it, CPU (WebAssembly) otherwise. No server is involved; the
weights are fetched from the model repo's `onnx/` folder and cached by the browser after the first load.

What it shows, for each message:

- the `choice` answer (which team) with the probability of every category and the confidence;
- the `noul` answer (ask a follow-up first?) as a probability;
- the same answers as a "smart if", with adjustable thresholds;
- the raw System One request and response, the tokens each question read and the time each forward pass took;
- presets taken from the held-out test split (Devanagari, Romanized and app-review style; look-alike categories,
  negation, out of scope, vague and multi-issue messages), a run over every preset, and your own extra questions.

## Run it locally

Any static file server works; WebGPU and the Cache API need `https://` or `localhost`.

```bash
python -m http.server 8000      # from the repository root
# open http://localhost:8000/demo/
```

`?model=<url>` loads the weights from another folder that holds a `manifest.json` (for example a local build:
`/demo/?model=http://localhost:8000/build/webgpu/hub/onnx`).

## Hosting

The `demo/` folder is the whole site: `index.html`, `app.js`, `kev.js`, `style.css`, `coi-sw.js` and `data/`. Copy it to any static
host (GitHub Pages, Netlify, an S3 bucket). The page loads ONNX Runtime Web and the tokenizer from jsDelivr and the
weights from Hugging Face, which both allow cross-origin requests.

Measured speed: about 0.4-0.5 s per message (both questions) on WebGPU with an integrated GPU, and about
7 s on the CPU with 8 WebAssembly threads. The first load downloads 707 MB of weights; later visits read them from the
browser cache.

The CPU backend needs cross-origin isolation for more than one thread, and GitHub Pages cannot send the
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers. `coi-sw.js` is a small service worker that adds
them to the page's own files (the page reloads once on the first visit). Browsers that do not support
`Cross-Origin-Embedder-Policy: credentialless` run the CPU backend on one thread. WebGPU does not depend on any of this.

## How the browser model is built

Kev is a decision model: a Qwen3.5 backbone with a LoRA adapter and a small pointer head. The browser runs the backbone
as ONNX and the head in JavaScript:

1. `tools/merge_checkpoint.py` folds the adapter into the base model and writes the head as raw float32
   (`head.bin`, `head.json`, with the calibration temperature).
2. The [ONNX Runtime GenAI model builder](https://github.com/microsoft/onnxruntime-genai) exports the merged model
   without its language-model head, so the graph outputs hidden states (`exclude_lm_head=true`), for the WebGPU
   execution provider. The same graph runs on ONNX Runtime's CPU kernels.
3. `tools/package_model.py` lays out the `onnx/` folder and writes `manifest.json` (head shape, files, sizes and each
   graph's inputs, so the page can build empty past-state tensors).
4. `tools/parity.py` scores an export the way the page does (one causal row per question, read out at `<decide>` and
   every `</opt>`) and compares it with another export or with Kev's own predictions.

`kev.js` mirrors Kev's request handling: options are `name: description`, yes/no questions are `[no, yes]`, caller text
cannot produce delimiter tokens, and each question is the row
`<state> message <q> instruction (<opt> option </opt>)... <decide>`, with Kev's delimiters mapped to the reserved Qwen
tokens it uses.

```bash
uv sync --extra kev
uv run python demo/tools/merge_checkpoint.py --run sm079/nepkev --out build/webgpu/merged

# the rest in a separate environment with onnxruntime-genai, onnx and onnxruntime
B="python -m onnxruntime_genai.models.builder -i build/webgpu/merged/hf --extra_options exclude_lm_head=true exclude_mtp=true"
$B -o build/webgpu/onnx-fp32-cpu -p fp32 -e cpu            # reference
$B -o build/webgpu/onnx-int8-webgpu -p int8 -e webgpu
python demo/tools/quantize_embedding.py --model build/webgpu/onnx-int8-webgpu --out build/webgpu/onnx-int8e4-webgpu

python demo/tools/parity.py --model build/webgpu/onnx-int8e4-webgpu --ref-model build/webgpu/onnx-fp32-cpu     --ref-cache build/webgpu/ref-fp32.json --head build/webgpu/merged --kev data/mix-v3-add/kev/test.jsonl --n 120

python demo/tools/package_model.py --head build/webgpu/merged --out build/webgpu/hub/onnx     --variant int8=build/webgpu/onnx-int8e4-webgpu --variant int8-fp16emb=build/webgpu/onnx-int8-webgpu     --recommend webgpu=int8 --recommend wasm=int8
hf upload <user>/nepkev build/webgpu/hub/onnx onnx
```

Quantization, scored with `parity.py` on 120 test messages (240 questions, 108 of them issue questions with a single
label) against the fp32 export, which matches the PyTorch checkpoint:

| export | download | answers that differ from fp32 | max abs. probability change | issue accuracy |
|---|---|---|---|---|
| fp32 | 2.9 GB | | | 0.972 |
| fp16 (first 40 questions) | 1.5 GB | 0 | 0.002 | |
| int8, fp16 embeddings | 1,055 MB | 1 | 0.09 | 0.972 |
| **int8, 4-bit embeddings** (default) | 707 MB | 3 | 0.13 | 0.963 |
| int4, k-quant with int8 sensitive layers | 867 MB | 13 | 0.83 | 0.917 |
| int4 | 489 MB | 16 | 0.74 | 0.917 |

4-bit weights cost this model about 5 points of issue accuracy, so the page offers the two int8 builds.

The demo's data files come from the taxonomy and the test split:

```bash
uv run python demo/tools/build_demo_data.py --test data/mix-v3-add/splits/test.jsonl
```

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

The `demo/` folder is the whole site: `index.html`, `app.js`, `kev.js`, `style.css` and `data/`. Copy it to any static
host (GitHub Pages, Netlify, an S3 bucket). The page loads ONNX Runtime Web and the tokenizer from jsDelivr and the
weights from Hugging Face, which both allow cross-origin requests.

The CPU fallback runs single-threaded unless the page is cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin`
and `Cross-Origin-Embedder-Policy: require-corp` or `credentialless`). GitHub Pages cannot set those headers, so on CPU a
message takes several seconds per question; on WebGPU it takes a fraction of a second.

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

# a separate environment with onnxruntime-genai, onnx and onnxruntime
python -m onnxruntime_genai.models.builder -i build/webgpu/merged/hf -o build/webgpu/onnx-fp32-cpu \
    -p fp32 -e cpu --extra_options exclude_lm_head=true exclude_mtp=true
python -m onnxruntime_genai.models.builder -i build/webgpu/merged/hf -o build/webgpu/onnx-fp16-webgpu \
    -p fp16 -e webgpu --extra_options exclude_lm_head=true exclude_mtp=true

python demo/tools/parity.py --model build/webgpu/onnx-fp16-webgpu --ref-model build/webgpu/onnx-fp32-cpu \
    --ref-cache build/webgpu/ref-fp32.json --head build/webgpu/merged --kev data/mix-v3-add/kev/test.jsonl --n 120

python demo/tools/package_model.py --head build/webgpu/merged --out build/webgpu/hub/onnx \
    --variant fp16=build/webgpu/onnx-fp16-webgpu --recommend webgpu=fp16 --recommend wasm=fp16
hf upload <user>/nepkev build/webgpu/hub/onnx onnx
```

The demo's data files come from the taxonomy and the test split:

```bash
uv run python demo/tools/build_demo_data.py --test data/mix-v3-add/splits/test.jsonl
```

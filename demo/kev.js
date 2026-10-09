// nepkev in the browser: the fine-tuned Kev-0.8B decision model on ONNX Runtime Web.
//
// Kev never generates text. Each typed question runs as one causal row through the backbone,
//   <|fim_prefix|> state <|fim_middle|> instruction (<|box_start|> option <|box_end|>)... <|fim_suffix|>
// and a small pointer head scores every option from the hidden states at <|fim_suffix|> (<decide>) and at each
// <|box_end|> (</opt>). One forward pass per question; the answer is a probability for every option.
//
// The module takes ONNX Runtime and the tokenizer class from the caller, so the same code runs in the page and in Node.

const STATE = "<|fim_prefix|>", Q = "<|fim_middle|>", OPT = "<|box_start|>", END_OPT = "<|box_end|>", DECIDE = "<|fim_suffix|>";
const SPECIAL_RE = /<\|([A-Za-z0-9_]+)\|>/g;   // caller text can never forge a delimiter token (kev.model.user_tokens)
export const MAX_TRAIN_STATE = 384;            // message tokens the model was trained with (longer still runs)

// ---------------------------------------------------------------- request shape (kev.api)

function render(v) { return v == null ? "" : String(v); }
function optionText(name, desc) { return desc == null || desc === "" ? name : `${name}: ${render(desc)}`; }

/** System One questions -> rows to score: {id, type, instr, options, keys}. Mirrors kev.api.to_record. */
export function toQuestions(questions) {
  return Object.entries(questions).map(([id, q]) => {
    if (q.type === "noul") {
      const c = q.criteria || {};
      return { id, type: "noul", instr: render(q.instructions), options: [optionText("no", c.false), optionText("yes", c.true)], keys: ["false", "true"] };
    }
    if (q.type === "choice") {
      return { id, type: "choice", instr: render(q.instructions), options: Object.entries(q.criteria).map(([k, v]) => optionText(k, v)), keys: Object.keys(q.criteria) };
    }
    throw new Error(`question ${id}: type ${q.type} is not supported by this demo`);
  });
}

/** Probabilities -> System One answers (kev.api.to_answers). */
export function toAnswers(scored) {
  const out = {};
  for (const { q, p } of scored) {
    if (q.type === "noul") { out[q.id] = { type: "noul", noul: round(p[1]) }; continue; }
    const best = argmax(p), K = p.length;
    out[q.id] = { type: "choice", choice: q.keys[best], confidence: round(K === 1 ? 1 : (p[best] - 1 / K) / (1 - 1 / K)),
                  probabilities: Object.fromEntries(q.keys.map((k, i) => [k, round(p[i])])) };
  }
  return out;
}

const round = (x) => Math.round(x * 1e4) / 1e4;
export const argmax = (a) => a.reduce((b, x, i) => (x > a[b] ? i : b), 0);

// ---------------------------------------------------------------- downloads

const CACHE = "nepkev-model-v1";

/** fetch -> Uint8Array with progress, through the Cache API when the page has one (reloads skip the download). */
export async function fetchBytes(url, onProgress = () => {}) {
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE).catch(() => null) : null;
  let res = cache ? await cache.match(url).catch(() => null) : null;
  const hit = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) { const b = new Uint8Array(await res.arrayBuffer()); onProgress(b.length, b.length, hit); return b; }
  const reader = res.body.getReader();
  let buf = total ? new Uint8Array(total) : null, chunks = [], got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buf && got + value.length <= buf.length) buf.set(value, got); else { if (buf) { chunks.push(buf.subarray(0, got)); buf = null; } chunks.push(value); }
    got += value.length;
    onProgress(got, total || got, hit);
  }
  if (!buf || got !== buf.length) {
    const out = new Uint8Array(got); let o = 0;
    for (const c of buf ? [buf.subarray(0, got)] : chunks) { out.set(c, o); o += c.length; }
    buf = out;
  }
  if (cache && !hit) {
    await cache.put(url, new Response(buf, { headers: { "content-type": "application/octet-stream", "content-length": String(buf.length) } })).catch(() => {});
  }
  return buf;
}

export async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

// ---------------------------------------------------------------- fp16

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

// ---------------------------------------------------------------- the model

export class KevRouter {
  /**
   * @param ort        onnxruntime-web (or -node) module
   * @param Tokenizer  @huggingface/tokenizers' Tokenizer class
   * @param base       URL of the model folder (holds manifest.json)
   * @param variant    a key of manifest.variants
   * @param ep         "webgpu" or "wasm" (every variant runs on both)
   */
  static async load({ ort, Tokenizer, base, variant, ep = "wasm", onProgress = () => {}, sessionOptions = {} }) {
    base = base.replace(/\/$/, "");
    const manifest = await fetchJSON(`${base}/manifest.json`);
    const v = manifest.variants[variant];
    if (!v) throw new Error(`unknown variant ${variant}`);
    const files = [
      ["tokenizer.json", manifest.files["tokenizer.json"]], ["tokenizer_config.json", manifest.files["tokenizer_config.json"]],
      ["head.bin", manifest.files["head.bin"]], ...v.files.map((f) => [`${v.dir}/${f.name}`, f.bytes]),
    ];
    const total = files.reduce((s, [, b]) => s + (b || 0), 0), done = {};
    const report = (name) => (got, _t, hit) => { done[name] = got; onProgress({ loaded: Object.values(done).reduce((a, b) => a + b, 0), total, file: name, cached: hit }); };
    const get = (name) => fetchBytes(`${base}/${name}`, report(name));
    const dec = new TextDecoder();
    const [tokJson, tokCfg, head, ...graph] = await Promise.all(files.map(([n]) => get(n)));
    const tokenizer = new Tokenizer(JSON.parse(dec.decode(tokJson)), JSON.parse(dec.decode(tokCfg)));
    onProgress({ loaded: total, total, file: "", stage: "compile" });
    const [model, ...external] = graph;
    const session = await ort.InferenceSession.create(model, {
      executionProviders: [ep], graphOptimizationLevel: "all",
      externalData: v.files.slice(1).map((f, i) => ({ path: f.name, data: external[i] })), ...sessionOptions,
    });
    return new KevRouter(ort, session, tokenizer, head, manifest, v);
  }

  constructor(ort, session, tokenizer, headBytes, manifest, variant) {
    Object.assign(this, { ort, session, tokenizer, manifest, variant });
    const h = manifest.head, d = h.hidden_size, dp = h.pointer_dim;
    const f = new Float32Array(headBytes.buffer, headBytes.byteOffset, headBytes.byteLength / 4);
    this.d = d; this.dp = dp;
    this.qw = f.subarray(0, dp * d); this.qb = f.subarray(dp * d, dp * d + dp);
    this.kw = f.subarray(dp * d + dp, 2 * dp * d + dp); this.kb = f.subarray(2 * dp * d + dp);
    this.scale = 1 / Math.sqrt(dp); this.temperature = h.temperature;
    this.ids = Object.fromEntries([STATE, Q, OPT, END_OPT, DECIDE].map((t) => [t, tokenizer.token_to_id(t)]));
    this.empty = {};   // zero past-state tensors, built once
    for (const inp of variant.inputs) {
      if (!inp.empty_shape) continue;
      const shape = inp.empty_shape;
      const n = shape.reduce((a, b) => a * b, 1);
      this.empty[inp.name] = new ort.Tensor(inp.dtype, inp.dtype === "float16" ? new Uint16Array(n) : new Float32Array(n), shape);
    }
    this.queue = Promise.resolve();
  }

  tokens(text) { return this.tokenizer.encode(text.replace(SPECIAL_RE, "<¦$1¦>"), { add_special_tokens: false }).ids; }

  row(stateIds, q) {
    const ids = [this.ids[STATE], ...stateIds, this.ids[Q], ...this.tokens(q.instr)], ends = [];
    for (const o of q.options) { ids.push(this.ids[OPT], ...this.tokens(o), this.ids[END_OPT]); ends.push(ids.length - 1); }
    ids.push(this.ids[DECIDE]);
    return { ids, ends };
  }

  async hidden(ids) {
    const n = ids.length, T = this.ort.Tensor;
    const pos = new BigInt64Array(3 * n);
    for (let r = 0; r < 3; r++) for (let i = 0; i < n; i++) pos[r * n + i] = BigInt(i);   // text-only mrope: three equal rows
    const feeds = { input_ids: new T("int64", BigInt64Array.from(ids, BigInt), [1, n]),
                    attention_mask: new T("int64", new BigInt64Array(n).fill(1n), [1, n]),
                    position_ids: new T("int64", pos, [3, 1, n]), ...this.empty };
    const out = await this.session.run(feeds, ["hidden_states"]);
    const t = out.hidden_states, data = t.data;
    const get = t.type === "float16" && data instanceof Uint16Array ? (i) => halfToFloat(data[i]) : (i) => Number(data[i]);
    return { get, dispose: () => t.dispose?.() };
  }

  pointer(h, at) {   // hidden state at row position `at` through one projection of the head
    const { d } = this, v = new Float32Array(d);
    for (let j = 0; j < d; j++) v[j] = h.get(at * d + j);
    return v;
  }

  project(w, b, x) {
    const { d, dp } = this, y = new Float32Array(dp);
    for (let i = 0; i < dp; i++) { let s = b[i]; const o = i * d; for (let j = 0; j < d; j++) s += w[o + j] * x[j]; y[i] = s; }
    return y;
  }

  /** -> {p, tokens, ms} for one question on one message. */
  async scoreOne(stateIds, q) {
    const { ids, ends } = this.row(stateIds, q);
    const t0 = performance.now();
    const h = await this.hidden(ids);
    const qv = this.project(this.qw, this.qb, this.pointer(h, ids.length - 1));
    const z = ends.map((e) => { const kv = this.project(this.kw, this.kb, this.pointer(h, e)); let s = 0; for (let i = 0; i < this.dp; i++) s += kv[i] * qv[i]; return s * this.scale / this.temperature; });
    h.dispose();
    const m = Math.max(...z), ex = z.map((x) => Math.exp(x - m)), sum = ex.reduce((a, b) => a + b, 0);
    return { p: ex.map((x) => x / sum), tokens: ids.length, ms: performance.now() - t0 };
  }

  /** Score a System One request {state, questions}. Calls are serialised (one session, one pass at a time). */
  decide(request) {
    const job = this.queue.then(async () => {
      const qs = toQuestions(request.questions), stateIds = this.tokens(render(request.state)), scored = [];
      const t0 = performance.now();
      for (const q of qs) scored.push({ q, ...(await this.scoreOne(stateIds, q)) });
      return { answers: toAnswers(scored), scored, stateTokens: stateIds.length + 1, ms: performance.now() - t0 };
    });
    this.queue = job.catch(() => {});
    return job;
  }
}

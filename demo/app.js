import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.min.mjs";
import { KevRouter, MAX_TRAIN_STATE, argmax, fetchJSON } from "./kev.js";

// ?model=<url of a folder holding manifest.json> serves the weights from elsewhere (e.g. a local build)
const params = new URLSearchParams(location.search);
const REPO = "sm079/nepkev";
const MODEL_BASE = (params.get("model") || `https://huggingface.co/${REPO}/resolve/main/onnx`).replace(/\/$/, "");

// one ONNX Runtime build per backend: the WebGPU build's CPU kernels are slower and lack some quantized ops
const ORT_URL = {
  webgpu: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort.webgpu.min.mjs",
  wasm: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort.wasm.min.mjs",
};
const THREADS = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
let ort = null, ortBackend = null;

async function runtime(backend) {
  if (ort && ortBackend !== backend) {   // two runtimes in one page do not mix: start over with the other one
    const u = new URL(location.href); u.searchParams.set("backend", backend); u.searchParams.set("load", "1"); location.assign(u); await new Promise(() => {});
  }
  if (!ort) {
    ort = await import(ORT_URL[backend]); ortBackend = backend;
    ort.env.wasm.numThreads = THREADS;
  }
  return ort;
}

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
};
const pct = (x) => `${(100 * x).toFixed(x >= 0.995 || x < 0.0005 ? 0 : 1)}%`;
const mib = (b) => `${Math.round(b / 2 ** 20)} MB`;

const KIND = {
  single: "clear", contrast: "look-alike categories", negation: "negation", corrected_statement: "corrects itself",
  "multi_issue:stated_priority": "two issues, priority given", out_of_scope: "out of scope", vague: "vague",
  "multi_issue:no_priority": "two issues, no priority", irrelevant_background: "irrelevant background", missing_details: "missing details",
};
const ORIGIN = { clean: "Devanagari", translated: "Romanized", "app-review": "app-review style" };

const S = { taxonomy: null, presets: [], manifest: null, router: null, backend: null, domain: "shop", preset: null, busy: false,
            last: null, extras: [], batchRunning: false };

// ---------------------------------------------------------------- setup

async function init() {
  [S.taxonomy, S.presets] = await Promise.all([fetchJSON("data/taxonomy.json"), fetchJSON("data/presets.json")]);
  renderDomains(); renderPresets(); bind();
  $("repo-link").href = params.get("model") ? MODEL_BASE : `https://huggingface.co/${REPO}`;
  try {
    S.manifest = await fetchJSON(`${MODEL_BASE}/manifest.json`);
  } catch (e) {
    setStatus("error", `Could not read the model manifest (${e.message})`); $("load").disabled = true; return;
  }
  if (["auto", "webgpu", "wasm"].includes(params.get("backend"))) $("backend").value = params.get("backend");
  await pickBackend();
  if (params.get("load") === "1" || await isCached()) load();
}

async function webgpuAvailable() {
  if (!navigator.gpu) return false;
  try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function pickBackend() {
  const gpu = await webgpuAvailable();
  const opt = $("backend").querySelector('option[value="webgpu"]');
  opt.disabled = !gpu; if (!gpu) opt.textContent = "WebGPU (not available)";
  const choice = $("backend").value;
  S.backend = choice === "auto" ? (gpu ? "webgpu" : "wasm") : choice;
  const sel = $("variant"), rec = S.manifest.recommended?.[S.backend];
  const keep = sel.value;
  sel.replaceChildren(...Object.entries(S.manifest.variants).map(([k, v]) => el("option", { value: k, text: `${v.label || k} · ${mib(v.bytes)}` })));
  sel.value = keep && S.manifest.variants[keep] && sel.dataset.touched ? keep : rec || Object.keys(S.manifest.variants)[0];
  updateLoadButton();
}

function variantFiles(name) {
  const v = S.manifest.variants[name];
  return [...Object.keys(S.manifest.files), ...v.files.map((f) => `${v.dir}/${f.name}`)].map((f) => `${MODEL_BASE}/${f}`);
}

async function isCached() {
  if (typeof caches === "undefined") return false;
  try {
    const c = await caches.open("nepkev-model-v1");
    return (await Promise.all(variantFiles($("variant").value).map((u) => c.match(u)))).every(Boolean);
  } catch { return false; }
}

async function updateLoadButton() {
  const v = S.manifest?.variants[$("variant").value];
  if (!v) return;
  const cached = await isCached();
  $("load").textContent = S.router ? "Reload" : cached ? "Load (cached)" : `Load model · ${mib(v.bytes + Object.values(S.manifest.files).reduce((a, b) => a + b, 0))}`;
  $("footnote").textContent = `${v.note || ""} Backend: ${S.backend === "webgpu" ? "WebGPU (GPU)" : `CPU via WebAssembly, ${THREADS} thread${THREADS > 1 ? "s" : ""}: expect a few seconds per question`}. Weights are cached by the browser after the first download.`;
}

function setStatus(kind, text) {
  $("status").className = `status ${kind}`; $("status-text").textContent = text;
}

async function load() {
  const variant = $("variant").value, backend = S.backend;
  $("load").disabled = true; $("route").disabled = true; $("batch").disabled = true;
  $("progress").hidden = false; $("progress-bar").style.width = "0";
  setStatus("busy", "Downloading…");
  try {
    S.router?.session?.release?.();
    S.router = null;
    const ort = await runtime(backend);
    S.router = await KevRouter.load({
      ort, Tokenizer, base: MODEL_BASE, variant, ep: backend,
      onProgress: ({ loaded, total, stage, cached }) => {
        $("progress-bar").style.width = `${(100 * loaded) / (total || 1)}%`;
        setStatus("busy", stage === "compile" ? `Preparing the model on ${backend === "webgpu" ? "the GPU" : "the CPU"}…` : `${cached ? "Reading cache" : "Downloading"} ${mib(loaded)} / ${mib(total)}`);
      },
    });
    setStatus("busy", "Warming up…");
    const t0 = performance.now();
    await S.router.decide({ state: "नमस्ते", questions: { q: S.taxonomy.domains.shop.clarification } });
    setStatus("ready", `Ready · ${backend === "webgpu" ? "WebGPU" : "CPU (WASM)"} · ${S.manifest.variants[variant].label || variant} · warm-up ${Math.round(performance.now() - t0)} ms`);
    $("route").disabled = false; $("batch").disabled = false;
    updateTokens();
  } catch (e) {
    console.error(e);
    if (backend === "webgpu") {
      setStatus("error", `WebGPU failed (${short(e)}). Switching to CPU…`);
      $("backend").value = "wasm"; await pickBackend(); $("progress").hidden = true; $("load").disabled = false;
      return load();
    }
    setStatus("error", `Could not load the model: ${short(e)}`);
  } finally {
    $("progress").hidden = true; $("load").disabled = false; updateLoadButton();
  }
}
const short = (e) => String(e?.message || e).split("\n")[0].slice(0, 160);

// ---------------------------------------------------------------- state panel

function renderDomains() {
  $("domains").replaceChildren(...Object.entries(S.taxonomy.domains).map(([k, d]) =>
    el("button", { class: k === S.domain ? "on" : "", role: "tab", "aria-selected": String(k === S.domain), onclick: () => setDomain(k) },
       d.name, el("span", { class: "ne", text: d.ne }))));
}

function setDomain(k) {
  if (k === S.domain) return;
  S.domain = k; S.preset = null; renderDomains(); renderPresets();
}

function renderPresets() {
  $("presets").replaceChildren(...S.presets.filter((p) => p.domain === S.domain).map((p) =>
    el("button", { class: `preset${S.preset === p ? " on" : ""}`, onclick: () => usePreset(p) },
       el("span", { class: "meta" }, el("span", { class: "pill kind", text: KIND[p.family.split(":")[0]] || KIND[p.family] || p.family }),
          el("span", { class: "pill", text: ORIGIN[p.origin] })),
       el("span", { class: "text", text: p.message }))));
}

function usePreset(p) {
  S.preset = p; $("message").value = p.message; renderPresets(); updateTokens();
  if (S.router && !S.busy) route();
}

function updateTokens() {
  const t = $("tokens"), text = $("message").value;
  if (!S.router || !text) { t.textContent = ""; return; }
  const n = S.router.tokens(text).length + 1;
  t.textContent = `${n} message tokens`;
  t.className = n > MAX_TRAIN_STATE ? "warn" : "muted";
  if (n > MAX_TRAIN_STATE) t.textContent += ` (trained on up to ${MAX_TRAIN_STATE})`;
}

// ---------------------------------------------------------------- questions

function request(message, domain) {
  const d = S.taxonomy.domains[domain], questions = { q1: d.issue, q2: d.clarification };
  S.extras.forEach((q, i) => { questions[`q${i + 3}`] = q; });
  return { state: message, questions };
}

function renderExtras() {
  $("extra-list").replaceChildren(...S.extras.map((q, i) =>
    el("div", { class: "extra-item" }, el("span", { class: "qtype", text: q.type }), el("span", { text: q.instructions + (q.type === "choice" ? `  [${Object.keys(q.criteria).join(", ")}]` : "") }),
       el("button", { title: "Remove", "aria-label": "Remove question", onclick: () => { S.extras.splice(i, 1); renderExtras(); }, text: "×" }))));
}

// ---------------------------------------------------------------- routing

async function route() {
  const message = $("message").value.trim();
  if (!message || !S.router) return;
  if (S.preset && S.preset.message !== message) { S.preset = null; renderPresets(); }
  S.busy = true; $("route").disabled = true; $("route").textContent = "Routing…";
  try {
    const req = request(message, S.domain);
    const res = await S.router.decide(req);
    S.last = { req, res, preset: S.preset, domain: S.domain };
    renderResult();
  } catch (e) {
    console.error(e); setStatus("error", `Inference failed: ${short(e)}`);
  } finally {
    S.busy = false; $("route").disabled = false; $("route").textContent = "Route";
  }
}

function bars(keys, p, gold) {
  const order = keys.map((k, i) => i).sort((a, b) => p[b] - p[a]), top = order[0];
  return el("div", { class: "bars" }, order.map((i) =>
    el("div", { class: `row${i === top ? " lead" : ""}`, title: keys[i] },
       el("span", { class: `k${keys[i] === gold ? " gold" : ""}`, text: keys[i] }),
       el("span", { class: "track" }, el("span", { class: "fill", style: `width:${(100 * p[i]).toFixed(2)}%` })),
       el("span", { class: "v", text: pct(p[i]) }))));
}

function expectTag(ok, text) { return el("span", { class: `expect ${ok == null ? "na" : ok ? "ok" : "no"}`, text }); }

function renderResult() {
  const { req, res, preset, domain } = S.last, d = S.taxonomy.domains[domain];
  $("empty").hidden = true;
  const cards = res.scored.map(({ q, p }) => {
    const a = res.answers[q.id];
    if (q.id === "q1") {
      const choice = a.choice, gold = preset?.issue;
      let tag = null;
      if (preset) tag = gold ? expectTag(gold === choice, gold === choice ? "matches label" : `label: ${gold}`)
                             : expectTag(null, `no single answer (${preset.issues_all.join(" + ") || "vague"})`);
      return el("div", { class: "card" },
        el("div", { class: "card-head" }, el("span", { class: "qtype" }, el("b", { text: "q1 · choice" }), " which team?"), tag),
        el("div", { class: "answer", text: choice }),
        el("div", { class: "answer-ne", text: d.names_ne[choice] || "" }),
        el("div", { class: "desc", text: d.issue.criteria[choice] }),
        el("div", { class: "conf", text: `confidence ${a.confidence.toFixed(3)} · p ${pct(p[argmax(p)])}` }),
        bars(q.keys, p, gold));
    }
    if (q.type === "noul") {
      const yes = p[1], gold = q.id === "q2" ? preset?.clarification : undefined;
      const label = q.id === "q2" ? "q2 · noul" : `${q.id} · noul`;
      return el("div", { class: "card" },
        el("div", { class: "card-head" }, el("span", { class: "qtype" }, el("b", { text: label }), q.id === "q2" ? " ask a follow-up first?" : ` ${req.questions[q.id].instructions}`),
           gold == null ? null : expectTag((yes >= 0.5) === gold, `label: ${gold ? "yes" : "no"}`)),
        el("div", { class: "gauge" }, el("span", { class: "track" }, el("span", { class: "fill", style: `width:${(100 * yes).toFixed(2)}%` })),
           el("span", { class: "big", text: pct(yes) })),
        el("div", { class: "desc", text: yes >= 0.5 ? `yes: ${req.questions[q.id].criteria?.true || ""}` : `no${req.questions[q.id].criteria?.false ? `: ${req.questions[q.id].criteria.false}` : ""}` }));
    }
    return el("div", { class: "card" },
      el("div", { class: "card-head" }, el("span", { class: "qtype" }, el("b", { text: `${q.id} · choice` }), ` ${req.questions[q.id].instructions}`)),
      el("div", { class: "answer", text: a.choice }), el("div", { class: "conf", text: `confidence ${a.confidence.toFixed(3)}` }), bars(q.keys, p));
  });
  $("cards").replaceChildren(...cards);
  showView(document.querySelector("[data-view].on").dataset.view);

  $("timing").hidden = false;
  $("timing").replaceChildren(
    ...res.scored.map(({ q, tokens, ms }) => el("span", {}, `${q.id} `, el("b", { text: `${tokens}` }), " tokens in · ", el("b", { text: `${Math.round(ms)} ms` }))),
    el("span", {}, "total ", el("b", { text: `${Math.round(res.ms)} ms` })),
    el("span", {}, "output tokens ", el("b", { text: "0" })));

  $("json-req").textContent = JSON.stringify(req, null, 2);
  $("json-res").textContent = JSON.stringify({ answers: res.answers, usage: { input_tokens: res.scored.reduce((s, x) => s + x.tokens, 0), output_tokens: 0 },
                                                timing_ms: Object.fromEntries(res.scored.map(({ q, ms }) => [q.id, Math.round(ms)])) }, null, 2);
  $("smart").hidden = false; renderCode();
}

function renderCode() {
  if (!S.last) return;
  const a = S.last.res.answers, conf = +$("conf").value, clar = +$("clar").value;
  $("conf-out").textContent = conf.toFixed(2); $("clar-out").textContent = clar.toFixed(2);
  const ask = a.q2.noul >= clar, sure = a.q1.confidence >= conf;
  const branch = ask ? 0 : sure ? 1 : 2;
  const L = (i, html) => `<span class="line ${i === branch ? "on" : "off"}">${html}</span>`;
  $("code").innerHTML = [
    `<span class="c line">// const { q1, q2 } = await kev.decide({ state, questions })</span>`,
    `<span class="c line">// q1.choice = "${a.q1.choice}", q1.confidence = ${a.q1.confidence.toFixed(3)}, q2.noul = ${a.q2.noul.toFixed(3)}</span>`,
    L(0, `if (q2.noul &gt;= <span class="hit">${clar.toFixed(2)}</span>) askCustomer("के समस्या हो, अलि खुलाएर भन्नुहोस् न?")`),
    L(1, `else if (q1.confidence &gt;= <span class="hit">${conf.toFixed(2)}</span>) routeTo("<span class="hit">${a.q1.choice}</span>")`),
    L(2, `else sendToHumanTriage()`),
  ].join("");
}

// ---------------------------------------------------------------- batch

async function runBatch() {
  if (!S.router || S.batchRunning) return;
  const list = $("batch-scope").value === "all" ? S.presets : S.presets.filter((p) => p.domain === S.domain);
  S.batchRunning = true; $("batch").disabled = true; $("route").disabled = true;
  const tbody = $("batch-table").querySelector("tbody"); tbody.replaceChildren(); $("batch-table").hidden = false;
  let right = 0, scored = 0, clarRight = 0, totalMs = 0;
  const t0 = performance.now();
  try {
    for (const [i, p] of list.entries()) {
      $("batch-summary").textContent = `Routing ${i + 1}/${list.length}…`;
      const res = await S.router.decide(request(p.message, p.domain));
      const a = res.answers, ok = p.issue ? a.q1.choice === p.issue : null, cOk = (a.q2.noul >= 0.5) === p.clarification;
      if (ok != null) { scored++; right += ok; }
      clarRight += cOk; totalMs += res.ms;
      tbody.append(el("tr", {},
        el("td", { class: "msg" }, el("div", { text: p.message, title: p.message }), el("span", { class: "pill", text: `${p.domain} · ${ORIGIN[p.origin]}` })),
        el("td", { class: "id", text: p.issue || "(none)" }),
        el("td", { class: "id", text: a.q1.choice }),
        el("td", { class: "num", text: a.q1.confidence.toFixed(2) }),
        el("td", { class: "num", text: `${pct(a.q2.noul)}${p.clarification ? " ●" : ""}` }),
        el("td", { class: "num", text: Math.round(res.ms) }),
        el("td", { class: `mark ${ok == null ? "" : ok ? "ok" : "no"}`, text: ok == null ? "–" : ok ? "✓" : "✗" })));
    }
    const wall = performance.now() - t0;
    $("batch-summary").textContent = `${list.length} messages, ${2 * list.length + S.extras.length * list.length} forward passes in ${(wall / 1000).toFixed(1)} s ` +
      `(${Math.round(totalMs / list.length)} ms per message). Category right on ${right}/${scored} with a single label; ` +
      `clarification right on ${clarRight}/${list.length} (● = should ask). The model's full test-set numbers are on the model card.`;
  } catch (e) {
    console.error(e); $("batch-summary").textContent = `Stopped: ${short(e)}`;
  } finally {
    S.batchRunning = false; $("batch").disabled = false; $("route").disabled = false;
  }
}

// ---------------------------------------------------------------- events

function showView(v) {
  document.querySelectorAll("[data-view]").forEach((b) => { b.classList.toggle("on", b.dataset.view === v); b.setAttribute("aria-selected", String(b.dataset.view === v)); });
  if (!S.last) return;
  $("cards").hidden = v !== "cards"; $("json").hidden = v !== "json";
}

function bind() {
  $("load").addEventListener("click", load);
  $("backend").addEventListener("change", async () => { await pickBackend(); });
  $("variant").addEventListener("change", () => { $("variant").dataset.touched = "1"; updateLoadButton(); });
  $("route").addEventListener("click", route);
  $("batch").addEventListener("click", runBatch);
  $("message").addEventListener("input", updateTokens);
  $("message").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); route(); } });
  $("conf").addEventListener("input", renderCode); $("clar").addEventListener("input", renderCode);
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  $("extra-type").addEventListener("change", () => { $("extra-opts").hidden = $("extra-type").value !== "choice"; });
  $("extra-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = $("extra-q").value.trim(); if (!text) return;
    if ($("extra-type").value === "choice") {
      const opts = $("extra-opts").value.split(",").map((s) => s.trim()).filter(Boolean);
      if (opts.length < 2) { $("extra-opts").focus(); return; }
      S.extras.push({ type: "choice", instructions: text, criteria: Object.fromEntries(opts.map((o) => [o, null])) });
    } else S.extras.push({ type: "noul", instructions: text });
    $("extra-q").value = ""; $("extra-opts").value = ""; renderExtras();
  });
}

init().catch((e) => { console.error(e); setStatus("error", short(e)); });

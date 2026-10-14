# nepkev: Nepali support-message routing on Kev

`nepkev` fine-tunes [Kev](https://github.com/jaredpalmer/kev) 0.8B, a small decision model, to route Nepali
customer-support messages for three demo services: an online shop, a digital wallet and a bank. For each message it
answers two questions:

1. **issue**: which of the service's categories the problem belongs to (`other` for clear requests outside the list);
2. **clarification**: whether the agent has to ask a follow-up question first (vague messages, several problems with
   no stated priority).

Messages are Nepali in Devanagari or Romanized script, often mixed with English. Instructions and category
descriptions stay English; only the customer message is Nepali. The router only routes: it does not resolve
transactions, authenticate anyone or give financial advice. The taxonomy is a demo (`configs/taxonomy/demo-1.yaml`,
notes in [docs/taxonomy-notes.md](docs/taxonomy-notes.md)), not any real institution's.

This repository holds the data preparation for the generated training data and the training and evaluation code.
Full results and the findings behind each design choice are in [docs/results.md](docs/results.md).

## Best run so far: mix-v3-add

Kev-0.8B fine-tuned for 2 epochs on 7,400 records:

| part | records | what it is |
|---|---|---|
| clean synthetic | 2,378 | LLM-written casual Devanagari, one per scenario |
| translated | 2,369 | code-derived Romanized twins of those, English mixed in |
| app-review style | 2,274 | the same scenarios written again by the LLM the way people type app reviews (85% Romanized, 15% Devanagari) |
| external | 379 | labelled Nepali app-store reviews (training only) |

Every category has about the same share of each domain's training labels (6.7-10.3%, `other` 7.0-8.0%).

Test results on the generated test set (889 records, never trained on; 95% CIs from a scenario-group bootstrap):

| predictor | issue accuracy | macro-F1 | issue NLL | ECE | clarification F1 |
|---|---|---|---|---|---|
| keyword rules | 0.619 | | | | 0.306 |
| Kev-0.8B untouched | 0.425 | | | | 0.182 |
| **Kev-0.8B fine-tuned (calibrated)** | **0.966** [0.953, 0.980] | 0.965 | 0.122 | 0.021 | 0.874 [0.797, 0.933] |

Fine-tuning adds +0.541 issue accuracy [0.496, 0.588] over the untouched model on the same messages. By part of the
test set (issue accuracy):

| | untouched | fine-tuned |
|---|---|---|
| clean Devanagari | 0.301 | 0.985 |
| Romanized | 0.420 | 0.952 |
| app-review style | 0.557 | 0.962 |

All bias checks pass for the fine-tuned model (no category over-predicted, `other` predicted 1.03x its true share with
precision 0.96); the untouched model sends most messages to `other` (5x its share) and asks to clarify almost always.

On real Nepali app reviews (215 scored, labels from an LLM, not human-checked) the same model gets 0.600 issue
accuracy [0.535, 0.660], against 0.237 untouched and 0.335 for the rules. Most errors send a real problem to `other`
or swap one money category for another. Details in [docs/real-text.md](docs/real-text.md).

Checkpoint: [`sm079/nepkev`](https://huggingface.co/sm079/nepkev).

A static browser demo in [`demo/`](demo/README.md) runs this checkpoint client-side with ONNX Runtime Web (WebGPU,
or CPU through WebAssembly): preset test messages, your own messages and the raw request and response.

## Setup

```bash
uv sync                   # core: generation, validation, splits, export, evaluation
uv sync --extra hub       # + huggingface_hub, for push-hf
uv sync --extra kev --extra service --extra rag   # + the API and the RAG layer (see below)
uv run pytest             # network-free tests; the API's need Postgres (see Run as an API), one needs the kev extra
```

API steps need `ANTHROPIC_API_KEY` (generation, verification) or `RUNPOD_API_KEY` (training), from the
environment or a `.env` file (copy `.env.example`).
Every command that calls an API refuses to run without `--yes` and skips work already on disk, so a run directory whose model
outputs are all present rebuilds offline without an API key.

A network-free end-to-end check with mock messages:

```bash
uv run nepkev pipeline --config fixture-dryrun --out data/dryrun
```

## Run as an API

`nepkev.service` serves the checkpoint over HTTP (FastAPI) and logs every routed message to Postgres.

```bash
docker compose up --build        # Postgres, one-shot migration and ingest, the API on :8000
curl localhost:8000/ready        # 503 while the model loads, then {"ready":true,...}
```

The first start downloads the pinned checkpoint and its base model (about 2 GB) into a volume; two cold starts took 2.5
and 7 minutes to become ready, mostly downloading. Later starts only load the model (about 40 seconds).

```bash
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
     -d '{"text": "OTP aaudaina login garna sakdina", "channel": "wallet"}'
```

```json
{"request_id": "75c786e3e04743df81c4ff39c8aeeaf5", "channel": "wallet", "intent": "login_otp", "confidence": 0.9999,
 "needs_clarification": false, "clarification_reason": null, "clarifying_question": null,
 "model_version": "sm079/nepkev@8daf1b3", "latency_ms": 3625.0}
```

Devanagari works the same way. On Windows shells, send it from a UTF-8 file (`--data-binary @body.json`) so the
terminal does not re-encode it:

```bash
echo '{"text": "मेरो डेबिट कार्ड हरायो, ब्लक गरिदिनुस्", "channel": "bank"}' > body.json
curl -X POST localhost:8000/route -H 'Content-Type: application/json' --data-binary @body.json
# -> "intent": "card_issue", "confidence": 0.996, "needs_clarification": false
```

When the model is unsure, the API asks rather than guesses: `needs_clarification` is true when the model judges the
message unclear (clarification probability >= 0.5, the evaluated decision rule) or when the calibrated probability of
the top category is below 0.5 (`NEPKEV_MIN_CONFIDENCE`). The question is in the customer's script. The model misses
mobile top-up wording, for example, and says so through its confidence:

```json
{"text": "mero paisa katyo tara recharge bhayena"}
-> {"intent": "identity_verification", "confidence": 0.3796, "needs_clarification": true,
    "clarification_reason": "low_confidence",
    "clarifying_question": "tapainko samasya yimadhye kun ho: KYC/pahichan pramanikaran samasya, ki paisa pathaiyo tara paune lai pugena? ali khulaera bhannuhos na.", ...}
```

| endpoint | what it does |
|---|---|
| `POST /route` | `{"text", "channel"?}` -> intent, calibrated confidence, clarification decision and question. Text is 1-2,000 characters (422 otherwise); `channel` is `shop`, `wallet` or `bank` and defaults to `NEPKEV_DEFAULT_CHANNEL`. An optional `Idempotency-Key` header replays the stored response for the same body (header `Idempotent-Replayed: true`) and rejects the same key with a different body (422). |
| `POST /answer` | `{"text", "channel"?}` -> a cited answer from the knowledge base, or a refusal; see [RAG layer](#rag-layer-cited-answers) |
| `GET /health` | liveness: the process is up |
| `GET /ready` | readiness: model loaded and database reachable (503 otherwise); `answer` says whether `/answer` can serve |
| `GET /stats?hours=24` | requests, clarification rate, p50/p95 model latency and counts per channel and intent over the window, computed in SQL |

Each request is logged to `route_requests` with a SHA-256 of the text, its script (Devanagari, Romanized or mixed),
the routing and the model version. Raw text is stored only with `NEPKEV_LOG_TEXT=true`. Settings are in
[`.env.example`](.env.example).

### Latency on CPU

`scripts/bench_api.py` sends the 33 held-out test messages of the browser demo three times, one request at a time, to
the container on a 16-thread laptop CPU (x86-64 with native bf16). Model time is what the API reports in
`latency_ms`; round trip is what the client measured.

| precision | model p50 | model p95 | round trip p50 | round trip p95 | matches the test label |
|---|---|---|---|---|---|
| fp32 (default, the evaluated path) | 3,694 ms | 4,561 ms | 3,917 ms | 4,826 ms | 26/27 |
| bf16 (`NEPKEV_DTYPE=bf16`) | 2,661 ms | 3,820 ms | 2,874 ms | 4,135 ms | 26/27 |

Each request is one forward pass of 347-441 tokens (median 392) for these messages, of which the message itself is 27-102 (median 57); the rest is the two questions and the category descriptions. bf16 is only
faster on CPUs with native bf16 support; elsewhere keep fp32.

### Architecture

```mermaid
flowchart LR
    client([client]) -->|POST /route| api
    subgraph container[api container]
        api[FastAPI app] --> decide[clarification policy<br/>nepkev.service.clarify]
        api --> router[KevRouter<br/>nepkev.router]
        router --> kev[upstream Kev<br/>pointer head, calibrated]
    end
    hub[(Hugging Face Hub<br/>sm079/nepkev @ pinned commit)] -.->|first start| router
    api -->|route_requests| pg[(Postgres)]
    api -->|GET /stats: SQL percentiles| pg
    migrate[alembic upgrade head] -->|one-shot| pg
```

The router sends the model exactly the request the evaluation scored (serving question wording, canonical category
order; a test checks it against the evaluation export), so the served probabilities are the evaluated ones.

### Without Docker

```bash
uv sync --extra kev --extra service --extra rag
export NEPKEV_DATABASE_URL=postgresql+psycopg://nepkev:nepkev@localhost:5432/nepkev
uv run alembic upgrade head
uv run python -m nepkev.rag.ingest          # with --extra rag; only needed for /answer
uv run python -m nepkev.service             # http://127.0.0.1:8000/docs
```

The API tests use a stub router and a real Postgres, which they wipe: point `NEPKEV_TEST_DATABASE_URL` at a scratch
database to run them (they are skipped otherwise). `uv run pytest -m slow` also runs the real checkpoint.

## RAG layer: cited answers

`POST /answer` answers a customer's question from a small knowledge base and shows its sources. The knowledge base in
[`kb/`](kb/README.md) is **synthetic**: 58 short help articles (29 topics, each in English and Nepali) about refunds,
failed top-ups, KYC, limits, card blocks and deliveries for the same fictional shop, wallet and bank. Every policy,
fee and time in it was written for this demo; none describes a real company.

```mermaid
flowchart LR
    q([question + channel]) --> router[KevRouter<br/>intent]
    q --> embed[multilingual-e5-small<br/>query embedding]
    embed --> pg[(pgvector kb_chunks<br/>HNSW, filtered by domain)]
    pg -->|top 5 chunks| gate{best similarity<br/>>= threshold?}
    gate -->|no| refuse[fixed refusal<br/>no model call]
    gate -->|yes| llm[answer model<br/>prompt answer_v1, JSON schema]
    llm -->|declines| refuse
    llm --> check{every quote<br/>verbatim in a<br/>retrieved chunk?}
    check -->|keeps valid citations;<br/>none left| refuse
    check --> answer([answer + citations])
```

1. The router gives the intent; the channel picks the domain, and retrieval reads only that domain's chunks.
2. If the best chunk is less similar than the threshold in `configs/rag.yaml`, the answer is a fixed refusal in the
   question's language, with no model call.
3. Otherwise the answer model (Claude through the Anthropic API, behind a small interface with a fake for the tests)
   gets the question and the chunks, with the versioned prompt
   [`configs/prompts/answer_v1.md`](configs/prompts/answer_v1.md). It must answer in the question's script, use only
   the chunks, and quote each one it relies on. It may decline.
4. The service keeps only citations that quote a retrieved chunk verbatim; an answer left without one is refused.

```bash
docker compose up --build       # also runs the ingest step; /answer needs ANTHROPIC_API_KEY in .env
curl -X POST localhost:8000/answer -H 'Content-Type: application/json' \
     -d '{"text": "mero paisa katyo tara recharge bhayena", "channel": "wallet"}'
```

```json
{"channel": "wallet", "intent": "identity_verification", "refused": false, "refusal_reason": null,
 "answer": "Recharge fail bhaera pani wallet bata paisa katiyo bhane 24 ghanta bhitra aafai wallet ma firta aauchha. Top-up Pending dekhaayo bhane pheri prayas garnu aghi badhima 2 ghanta parkhanuhos, natra paisa dui patak katna sakchha. ...",
 "citations": [{"chunk_id": "wallet-failed-topup-ne#27d2d2caeea8", "doc_id": "wallet-failed-topup-ne",
                "quote": "मोबाइल रिचार्ज वा डाटा प्याक टप-अप फेल भएर पनि वालेटबाट पैसा काटिएमा २४ घण्टाभित्र आफैं वालेटमा फिर्ता हुन्छ।"}, ...],
 "retrieval_scores": [{"chunk_id": "wallet-failed-topup-ne#27d2d2caeea8", "doc_id": "wallet-failed-topup-ne", "score": 0.8328}, ...],
 "prompt_version": "answer_v1", "answer_model": "claude-opus-5-5",
 "latency_ms": {"route": 4119.5, "retrieve": 361.0, "generate": 7555.6, "total": 12036.7}}
```

Retrieval found the right article for the message the router alone was unsure of (the intent above is the router's
low-confidence guess). An out-of-scope question ("What interest rate do you give on a 1-year fixed deposit?", bank)
came back `"refused": true, "refusal_reason": "low_similarity"` after 60 ms of retrieval and no model call.

```bash
uv run python -m nepkev.rag.ingest      # idempotent: embeds only new or edited chunks, deletes removed ones
uv run python -m nepkev.rag.eval        # rebuilds eval/report.md from cached model outputs; --yes to call the API
uv run python -m nepkev.rag.compare     # the embedding model comparison below
```

### Embedding model

Three open multilingual models were compared on the evaluation questions, in memory
([`eval/embedders.md`](eval/embedders.md)). On dev:

| model | dim | recall@1 | recall@5 | MRR | separates must-refuse (AUC) | ms per query (CPU) |
|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 0.769 | 1.000 | 0.873 | 0.992 | 23 |
| `intfloat/multilingual-e5-base` | 768 | 0.731 | 0.923 | 0.836 | 1.000 | 48 |
| `BAAI/bge-m3` | 1024 | 0.885 | 1.000 | 0.920 | 0.915 | 182 |

`multilingual-e5-small` is used. It ties BGE-M3 on recall@5, which is what the answer model sees, separates
must-refuse questions better, and is 8x faster and a fifth of the size. BGE-M3 ranks the answering article first
more often (also on test: recall@1 0.872 vs 0.795), so it is the upgrade to try if the top chunk matters more.

### Results

From [`eval/report.md`](eval/report.md), produced by `python -m nepkev.rag.eval`. The question set
([`eval/rag_set.jsonl`](eval/rag_set.jsonl)) has 126 hand-written questions in English, Devanagari and Romanized
Nepali, 22 of them out of scope (not covered by the knowledge base, or not support questions). A fixed third (dev, 31)
is used only to choose the threshold; the numbers below are on the other 95 (test: 78 answerable, 17 must refuse).
95% intervals are bootstraps over topics, so a question's paraphrases in the three scripts are resampled together.

| retrieval (top 5 of the domain) | test |
|---|---|
| recall@5 | 0.936 [0.867, 0.987] |
| MRR@10 | 0.862 [0.784, 0.925] |
| recall@5 by script: English / Devanagari / Romanized | 1.000 / 0.917 / 0.885 |

| answers (`claude-opus-5-5`, prompt `answer_v1`) | test |
|---|---|
| refusal recall (out-of-scope questions refused) | 0.941 [0.789, 1.000] |
| refusal precision (refusals that were right) | 0.762 [0.562, 0.941] |
| answerable questions answered | 0.936 [0.880, 0.984] |
| citations quoting their chunk verbatim, as the model returned them (222) | 1.000 |
| answers citing an article that answers the question | 0.959 [0.913, 1.000] |
| answers judged supported by their cited chunks (74) | 0.973 [0.938, 1.000] |
| retrieval latency p50 / p95 | 29 / 43 ms |
| answer generation latency p50 / p95 | 5,303 / 8,988 ms |

The support judge is `claude-sonnet-5-5` with the rubric in `configs/prompts/judge_v1.md`. Its verdicts were checked
by hand on 22 answers, 20 taken at a fixed stride plus the 2 it did not call supported; all 22 agree
([`eval/judge_check.jsonl`](eval/judge_check.jsonl)). The 2 partly supported answers add advice the article does not
give. Answers and verdicts are committed (`eval/answers.jsonl`, `eval/judge.jsonl`).

**How the threshold was chosen.** On dev, the cut-off that keeps every answerable question and refuses as many
must-refuse ones as possible (0.8030). A first version maximised balanced accuracy on dev instead. On test it would
have refused 15% of answerable questions, so it was replaced: a question the gate wrongly refuses is lost, while one
it wrongly lets through still meets the model, which can decline it. With E5's narrow similarity range, the gate
alone separates poorly (test: refuses 6 of 17 out-of-scope questions and 3 of 78 answerable ones). Of the 16 correct
refusals, the gate made 6 and the model 10.

**Known failure cases** (all wrong refusal decisions on test are listed in the report):

- Romanized Nepali is the weakest script for retrieval (recall@5 0.885, recall@1 0.692 vs 0.929 for English).
  "rahadani le kyc garna milcha?" (passport for KYC) and "atm le paisa diyena, ... complain kasari garne?" were
  refused at the gate although their articles exist.
- The model declined two answerable Romanized questions whose answer was in the retrieved chunks: a 60,000 transfer
  against the 50,000 single-transaction limit, and limits for transfers between one's own accounts.
- "kathmandu ma aaja pani parcha?" (will it rain in Kathmandu today?), asked in the shop channel, was read as "will it
  arrive today too?" and answered with delivery times: correctly cited, but not what was asked.
- The questions were written by the same author as the articles, so they share its wording more than real customers
  would, and the set is small; the intervals above are wide for that reason.

**CI gate.** The `rag-gate` job ingests the knowledge base with the real embedding model into a pgvector service
container and runs `python -m nepkev.rag.eval --retrieval-only --check eval/baseline.json`, which fails when test
recall@5 (0.936) or the gate's refusal recall (0.353) drops below the committed baseline. It makes no API calls.

## Pipeline

All outputs go under `data/` and `runs/` (git-ignored).

### 1. Clean synthetic data

Scenarios are sampled first, with labels decided from scenario facts and taxonomy rules before any text exists. An
LLM then writes each scenario as one casual Devanagari message; a second model labels every message from its text
alone; a third settles disagreements; deterministic checks run last.

```bash
uv run nepkev estimate  --config pilot                          # token estimate
uv run nepkev scenarios --config pilot --out data/pilot         # 3,000 scenarios, balanced per category
uv run nepkev generate  --config pilot --out data/pilot --batch --yes   # Sonnet 5.5, Message Batches; run again to collect
uv run nepkev records   --config pilot --out data/pilot
uv run nepkev verify     --config pilot --out data/pilot --yes  # Haiku 5.5 labels every message from the text
uv run nepkev adjudicate --config pilot --out data/pilot --yes  # Sonnet 5.5 decides the verifier's disputes
uv run nepkev validate --in data/pilot/records.jsonl --out data/pilot/validated.jsonl
uv run nepkev romanize --config pilot --in data/pilot/validated.jsonl --out data/pilot/with_roman.jsonl
```

The best run's data: 3,000 generated, 2,964 accepted, 2,954 Romanized twins.

### 2. App-review-style data

The same accepted scenarios, written again in the style of app-store reviews: typos, chat spellings, ranting, swearing,
mostly Romanized. The generator sees 30 fixed example reviews (`configs/prompts/realstyle_style_examples.json`). Script is
assigned within each label bucket (15% Devanagari), so script says nothing about the label.

```bash
uv run nepkev realstyle-generate --config realstyle --out data/realstyle-sample --sample 40 --yes   # look at a sample first
uv run nepkev realstyle-generate --config realstyle --out data/realstyle --batch --yes             # run again to collect
uv run nepkev realstyle-records  --config realstyle --out data/realstyle
uv run nepkev verify     --config realstyle --out data/realstyle --yes
uv run nepkev adjudicate --config realstyle --out data/realstyle --yes
uv run nepkev validate --in data/realstyle/records.jsonl --out data/realstyle/validated.jsonl
```

The best run's data: 2,964 written, 2,831 accepted (3 label disputes lost, 122 script glitches such as stray Devanagari
vowel signs inside Romanized words, 7 not Nepali, 1 duplicate).

### 3. Split and export

```bash
uv run nepkev split --clean data/pilot/with_roman.jsonl --realstyle data/realstyle/validated.jsonl \
    [--external data/external] --out data/mix/splits
uv run nepkev export-kev --splits data/mix/splits --out data/mix/kev
```

Clean records are split 80/5/5/10 (train/dev/calib/test) by scenario group, after merging groups that share
near-identical text, so no paraphrase crosses splits. App-review-style records join the split their scenario already
has. `--external` takes a directory of additional labelled records (`train.jsonl`, `dev.jsonl`, `calib.jsonl` in the
record schema, `source.generator: "external"`); they are only trained on, never scored. The split prints each
category's share of the training labels; check it.

Training export: half the questions use a reworded instruction, 15% drop 1-3 wrong candidates, candidate order is
shuffled. The gold category is never removed (see [docs/results.md](docs/results.md#decisions)).

### 4. Train on RunPod

```bash
uv run nepkev remote-train --config kev-0.8b --kev-dir data/mix/kev --out runs/mix --yes
```

One RunPod pod (RTX 4090 first, hard deadline): scores the untouched model, fine-tunes 2 epochs from
the released checkpoint (each scored on dev), keeps the epoch with the lowest dev NLL, fits a temperature on calib,
then scores test and Kev's transfer-v4 suite. Results download before every termination, and the pod is always
terminated. Upstream pins are in `configs/upstream.yaml`; `configs/remote/kev-4b.yaml` runs the same job on Kev-4B.

### 5. Evaluate and publish

```bash
uv run nepkev baseline-rules --splits data/mix/splits --out runs/mix/rules
uv run nepkev evaluate --splits data/mix/splits --kev-dir data/mix/kev --runs runs/mix/runs \
    --rules runs/mix/rules --out runs/mix/report
uv run nepkev push-hf --runs runs/mix/runs --repo <user>/nepkev --report runs/mix/report/report.json \
    --data-note "<a sentence on the training data>" --code-url https://github.com/<user>/nepkev
```

The report compares keyword rules, the untouched model and the fine-tuned model: accuracy, macro-F1, NLL, Brier, ECE,
coverage at 5% error, clarification F1, routing at the calib threshold, slices by origin, domain, family and length,
bias checks, and retention on transfer-v4. `push-hf` uploads only the files Kev loads (adapter, its config, `head.pt`)
with a short model card built from the report, creates the repo private and refuses to write to a public one.

## Reproducing the best run

Steps 1-3 are deterministic given the stored model outputs (`raw.jsonl`, `verify.jsonl`, `adjudicator.jsonl`). This
code was checked against the best run: from those outputs it rebuilds the train, dev and calib files mix-v3-add trained
on byte for byte, and re-scores its predictions to identical numbers. Without the external records, the same commands
give the best run's mix minus those 379 records; that exact mix has not been trained yet.

## Layout

```
nepkev/                            the package
  taxonomy.py, scenarios.py        taxonomy, scenario sampling (labels before text)
  generation/                      prompt assembly, Anthropic provider (+ dry run), generate/verify/adjudicate, app-review style
  validate.py, romanize.py         deterministic checks; Devanagari -> Romanized twins
  split.py, export_kev.py          group split, training mix, Kev JSONL export
  baselines.py, evaluate.py        keyword router; metrics, bootstrap CIs, bias checks
  remote.py, train_eval.sh         RunPod job runner and the job it runs on the pod
  hub.py                           Hugging Face upload
  router.py                        serving-time routing of one message (loads the checkpoint once)
  service/                         HTTP API: FastAPI app, Postgres request log, Alembic migrations
  rag/                             knowledge-base chunking, embeddings, pgvector store, cited answers, evaluation
kb/                                synthetic help articles (English and Nepali) for /answer
eval/                              RAG question set, evaluation report, cached answers and judge verdicts
configs/                           taxonomy, generation, romanization, rules, remote, upstream pins
  prompts/                         generator and verifier prompts, the 30 style examples, answer and judge prompts
tests/                             tests; fixtures/ holds 84 hand-checked records
docs/                              results and findings, taxonomy notes
scripts/                           API latency benchmark
Dockerfile, docker-compose.yml     API image; Postgres + migration + API stack
```

## Limitations

- Real reviews score far lower than generated test data (0.600 vs 0.966); see [docs/real-text.md](docs/real-text.md).
- Generated messages always describe their problem clearly, because the prompt requires it. People often write half a
  complaint; the app-review-style data copies the surface of such text, not that vagueness.
- The app-review-style data was made in a limited way. Its style came from only 30 example reviews, a small fraction
  of the reviews available, the same 30 for every request. And it re-wrote the clean data's scenarios rather than
  starting from what such reviews are about, so it inherited their content: one clearly stated problem, sampled facts
  and the pilot's family mix, now in a messier voice. Each scenario also appears three times in training (Devanagari,
  Romanized twin, app-review style), which adds surface variety but no new situations.
- The temperature is fitted on calib, which comes from the same data as training (Kev's calibration script warns about
  this). Calibration on other data is likely worse: on transfer-v4, ECE went from 0.049 to 0.091.
- Romanization is rule-based. Some words come out in spellings typists might not use; meaning-changing errors are
  rejected by the record checks.

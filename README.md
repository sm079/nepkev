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

Checkpoint: [`sm079/nepkev`](https://huggingface.co/sm079/nepkev).

A static browser demo in [`demo/`](demo/README.md) runs this checkpoint client-side with ONNX Runtime Web (WebGPU,
or CPU through WebAssembly): preset test messages, your own messages and the raw request and response.

## Setup

```bash
uv sync                   # core: generation, validation, splits, export, evaluation
uv sync --extra hub       # + huggingface_hub, for push-hf
uv run pytest             # 94 network-free tests; one needs `uv sync --extra kev` and is skipped otherwise
```

API steps need `ANTHROPIC_API_KEY` (generation, verification) or `RUNPOD_API_KEY` (training), from the
environment or a `.env` file (copy `.env.example`).
Every command that calls an API refuses to run without `--yes` and skips work already on disk, so a run directory whose model
outputs are all present rebuilds offline without an API key.

A network-free end-to-end check with mock messages:

```bash
uv run nepkev pipeline --config fixture-dryrun --out data/dryrun
```

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
configs/                           taxonomy, generation, romanization, rules, remote, upstream pins
  prompts/                         generator and verifier prompts, the 30 style examples
tests/                             tests; fixtures/ holds 84 hand-checked records
docs/                              results and findings, taxonomy notes
```

## Limitations

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

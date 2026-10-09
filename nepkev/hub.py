"""Upload a run's selected checkpoint to a private Hugging Face model repo, with a short model card.

Only the files Kev loads are uploaded (the LoRA adapter, its config and head.pt; Kev takes the tokenizer from the base
model). The card is built from the evaluation report (`nepkev evaluate`'s report.json): the untouched and the
fine-tuned model on the same test messages. Repos are created private, and an existing public repo is refused rather
than written to.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors", "head.pt")
ADAPTED, UNTOUCHED, RULES = "Kev-0.8B adapted (calibrated)", "Kev-0.8B untouched", "keyword rules"
PAIRED = "_paired_adapted_minus_untouched"
ORIGIN_NAMES = {"synthetic": "clean Devanagari", "translated": "Romanized", "realstyle": "app-review style"}

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
language: [ne, en]
tags: [kev, lora, text-classification, nepali, customer-support, routing]
---

# {name}

[Kev](https://github.com/jaredpalmer/kev) {size} fine-tuned to route Nepali customer-support messages for three demo
services: an online shop, a digital wallet and a bank. Messages can be Devanagari or Romanized Nepali, often mixed with
English. For each message the model answers two questions:

- `q1`, **issue**: probabilities over the service's categories (`other` for clear requests outside the list);
- `q2`, **clarification**: the probability that the agent must ask a follow-up question first.

Instructions and category descriptions are English; only the message is Nepali. The categories are a demo taxonomy,
not any real institution's. The model only routes.
{code}
## Results

On a held-out test set of {n} generated messages (95% CI from a scenario-group bootstrap):

| | issue accuracy | clarification F1 |
|---|---|---|
{table}

{gain}By part of the test set (issue accuracy):

{origin_table}

Issue ECE {ece:.3f} after temperature calibration. {bias}

## Training

{data_note}
LoRA fine-tuning from `{kev_model}@{kev_rev}` with Kev `{kev_commit}`, {epochs} epochs (the epoch with the lowest dev
NLL is kept), temperature fitted on a calibration split.

## Use

A Kev checkpoint: `adapter_model.safetensors` is the LoRA adapter on `{base}@{base_rev}`,
`head.pt` holds the pointer head and the fitted temperature. With [Kev](https://github.com/jaredpalmer/kev) at `{kev_commit}`:

```bash
python -m kev.serve --run {repo_id}        # POST /v1/systemone
```
{request_format}"""

REQUEST_FORMAT = """
The request format (question wording and category descriptions per service) is in the code's
`configs/taxonomy/demo-1.yaml`.
"""


def model_card(report: dict, repo_id: str, remote_cfg, data_note: str, code_url: str | None = None) -> str:
    """report: report.json from `nepkev evaluate` (its test split is used)."""
    test = report["test"]
    a, base = test[ADAPTED], test.get(UNTOUCHED)
    name = repo_id.split("/")[-1]

    def row(label: str, title: str, bold: bool = False) -> str:
        r = test[label]
        acc, f1 = f"{r['issue']['accuracy']:.3f}", f"{r['clarification']['f1']:.3f}"
        if bold:
            ci, cf = a["ci95"]["issue_accuracy"], a["ci95"]["clarification_f1"]
            acc, f1 = f"**{acc}** [{ci[0]:.3f}, {ci[1]:.3f}]", f"**{f1}** [{cf[0]:.3f}, {cf[1]:.3f}]"
        return f"| {title} | {acc} | {f1} |"

    baselines = ((RULES, "keyword rules"), (UNTOUCHED, f"Kev-{_size(remote_cfg)}, untouched"))
    table = "\n".join([*(row(lab, t) for lab, t in baselines if lab in test), row(ADAPTED, f"**{name}**", bold=True)])

    gain = ""
    if d := test.get(PAIRED):
        gain = (f"Fine-tuning adds {d['issue_accuracy_delta']:+.3f} issue accuracy [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                f"over the untouched model on the same messages. ")

    def origin_acc(rep: dict | None, k: str) -> str:
        v = (rep or {}).get("slices", {}).get("origin", {}).get(k, {}).get("issue_accuracy")
        return "n/a" if v is None else f"{v:.3f}"
    origins = a["slices"]["origin"]
    order = [k for k in [*ORIGIN_NAMES, *sorted(set(origins) - set(ORIGIN_NAMES))]
             if origins.get(k, {}).get("issue_accuracy") is not None]
    origin_table = "\n".join(["| | untouched | fine-tuned |", "|---|---|---|",
                              *(f"| {ORIGIN_NAMES.get(k, k)} | {origin_acc(base, k)} | {origin_acc(a, k)} |" for k in order)])

    checks = a["bias"]["checks"]
    if all(c.get("pass") is not False for c in checks.values()):
        o = checks["other_share_ratio"]["value"], checks["other_precision"]["value"]
        bias = f"No category is over-predicted (`other` at {o[0]:.2f}x its true share, precision {o[1]:.2f})."
    else:
        bias = "Bias checks failed: " + ", ".join(k for k, c in checks.items() if c.get("pass") is False) + "."

    code = f"\nCode, taxonomy and training pipeline: [{code_url.split('://')[-1]}]({code_url}).\n" if code_url else ""
    return CARD.format(
        base=remote_cfg.kev_base, name=name, size=_size(remote_cfg), code=code, n=a["records"], table=table, gain=gain,
        origin_table=origin_table, ece=a["issue"]["ece"], bias=bias, data_note=data_note.rstrip(".") + ".",
        kev_model=remote_cfg.kev_model, kev_rev=remote_cfg.kev_revision[:7], kev_commit=remote_cfg.kev_commit[:7],
        epochs=remote_cfg.epochs, base_rev=remote_cfg.kev_base_revision[:7], repo_id=repo_id,
        request_format=REQUEST_FORMAT if code_url else "")


def _size(remote_cfg) -> str:
    return remote_cfg.kev_model.rsplit("-", 1)[-1].upper()        # jaredpalmer/kev-0.8b -> 0.8B


def push(run_dir: Path, repo_id: str, remote_cfg, report_json: Path, data_note: str, code_url: str | None = None) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    ckpt = run_dir / "selected"
    if missing := [f for f in CHECKPOINT_FILES if not (ckpt / f).exists()]:
        raise FileNotFoundError(f"{ckpt} is not a Kev LoRA checkpoint (missing {missing})")
    if api.repo_exists(repo_id) and not api.model_info(repo_id).private:
        raise RuntimeError(f"{repo_id} exists and is public; refusing to upload")
    card = model_card(json.loads(report_json.read_text(encoding="utf-8")), repo_id, remote_cfg, data_note, code_url)
    api.create_repo(repo_id, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        for f in CHECKPOINT_FILES:
            shutil.copy(ckpt / f, stage / f)
        (stage / "README.md").write_text(card, encoding="utf-8", newline="\n")
        commit = api.upload_folder(repo_id=repo_id, folder_path=stage, commit_message=f"Upload {run_dir.parent.name} checkpoint")
    return commit.commit_url if hasattr(commit, "commit_url") else str(commit)

from nepkev.hub import CHECKPOINT_FILES, PAIRED, model_card
from nepkev.remote import load_remote_config


def _rep(acc, f1, origins=None, other=(1.0, 0.95), passed=True):
    checks = {"share_ratio": {"pass": passed}, "other_share_ratio": {"pass": True, "value": other[0]},
              "recall": {"pass": True}, "other_precision": {"pass": True, "value": other[1]}, "clarify_rate": {"pass": True}}
    return {"records": 100, "issue": {"accuracy": acc, "ece": 0.02}, "clarification": {"f1": f1},
            "ci95": {"issue_accuracy": [acc - 0.01, acc + 0.01], "clarification_f1": [f1 - 0.05, f1 + 0.05]},
            "slices": {"origin": {k: {"issue_accuracy": v} for k, v in (origins or {}).items()}}, "bias": {"checks": checks}}


def test_model_card_compares_the_untouched_and_the_fine_tuned_model():
    report = {"test": {"keyword rules": _rep(0.6, 0.3),
                       "Kev-0.8B untouched": _rep(0.4, 0.2, {"synthetic": 0.3, "translated": 0.42, "realstyle": 0.55}),
                       "Kev-0.8B adapted (calibrated)": _rep(0.97, 0.88, {"realstyle": 0.9, "synthetic": 0.99, "translated": 0.95}),
                       PAIRED: {"issue_accuracy_delta": 0.57, "ci95": [0.52, 0.62]}}}
    card = model_card(report, "me/nepkev", load_remote_config("kev-0.8b"), "Generated Nepali messages",
                      "https://github.com/me/nepkev")
    assert card.startswith("---\nbase_model: Qwen/Qwen3.5-0.8B-Base\n") and "# nepkev" in card
    assert "| Kev-0.8B, untouched | 0.400 | 0.200 |" in card
    assert "| **nepkev** | **0.970** [0.960, 0.980] | **0.880** [0.830, 0.930] |" in card
    assert "Fine-tuning adds +0.570 issue accuracy [+0.520, +0.620] over the untouched model" in card
    rows = [line for line in card.splitlines() if line.startswith("| clean") or line.startswith("| Romanized")
            or line.startswith("| app-review")]
    assert rows == ["| clean Devanagari | 0.300 | 0.990 |", "| Romanized | 0.420 | 0.950 |", "| app-review style | 0.550 | 0.900 |"]
    assert "No category is over-predicted (`other` at 1.00x its true share, precision 0.95)." in card
    assert "transfer" not in card and "Generated Nepali messages." in card
    assert "--run me/nepkev" in card and "github.com/me/nepkev" in card and len(card.splitlines()) < 70


def test_model_card_without_baselines_or_code_link():
    report = {"test": {"Kev-0.8B adapted (calibrated)": _rep(0.8, 0.6, {"synthetic": 0.8}, passed=False)}}
    card = model_card(report, "me/x", load_remote_config("kev-0.8b"), "Data.")
    assert "Bias checks failed: share_ratio." in card and "| clean Devanagari | n/a | 0.800 |" in card
    assert "Fine-tuning adds" not in card and "Code, taxonomy" not in card


def test_only_loadable_files_are_uploaded():
    assert CHECKPOINT_FILES == ("adapter_config.json", "adapter_model.safetensors", "head.pt")

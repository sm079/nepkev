"""Nepali support-message routing on a fine-tuned Kev decision model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
PROMPTS = CONFIGS / "prompts"
FIXTURES = ROOT / "tests" / "fixtures"

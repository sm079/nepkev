"""Service configuration from NEPKEV_* environment variables (or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Channel = Literal["shop", "wallet", "bank"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEPKEV_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nepkev:nepkev@localhost:5432/nepkev"

    # The checkpoint: a Hub repo id pinned to a commit, or a local run directory (revision then unused).
    model: str = "sm079/nepkev"
    model_revision: str | None = "8daf1b3bc570ea94e13ead9f8f0a2f2092bb3a3c"
    device: str = "cpu"
    dtype: Literal["fp32", "bf16"] = "fp32"   # fp32 is the evaluated path; bf16 is faster on CPUs with native bf16
    torch_threads: int | None = None

    # A deployment serves one support inbox, so the channel is usually known; requests may override it.
    default_channel: Channel = "wallet"
    # Ask the customer before routing when the model says the message is unclear (p >= clarify_threshold, the
    # evaluated decision rule) or when the calibrated top-category probability is below min_confidence.
    clarify_threshold: float = Field(0.5, ge=0, le=1)
    min_confidence: float = Field(0.5, ge=0, le=1)

    # Raw message text is not stored unless asked for; the log keeps a SHA-256 of it.
    log_text: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()

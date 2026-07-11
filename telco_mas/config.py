"""Runtime configuration, loaded from environment / .env.

The LLM layer is OpenAI-compatible: the same settings drive OpenAI or DeepSeek
(or any compatible gateway) simply by changing ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:  # optional dependency — .env is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing is fine
    pass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


@dataclass(frozen=True)
class Settings:
    """Immutable view of the current runtime settings."""

    api_key: str | None
    base_url: str
    model: str
    temperature: float
    cache_enabled: bool
    cache_dir: str
    max_tool_iters: int
    request_timeout: float
    seed: int | None = None

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_label(self) -> str:
        host = self.base_url.lower()
        if "deepseek" in host:
            return "DeepSeek"
        if "openai" in host:
            return "OpenAI"
        return "OpenAI-compatible"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("OPENAI_API_KEY") or None,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TELCO_TEMPERATURE", "0.1")),
        cache_enabled=_as_bool(os.getenv("LLM_CACHE"), default=False),
        cache_dir=os.getenv("TELCO_CACHE_DIR", ".llm_cache"),
        max_tool_iters=int(os.getenv("TELCO_MAX_TOOL_ITERS", "6")),
        request_timeout=float(os.getenv("TELCO_REQUEST_TIMEOUT", "60")),
        seed=_as_optional_int(os.getenv("TELCO_LLM_SEED")),
    )

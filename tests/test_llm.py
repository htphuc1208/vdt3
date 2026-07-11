from telco_mas.config import Settings
from telco_mas.llm import ChatResponse, LLMClient, LLMError

import pytest


def test_llm_client_passes_per_call_temperature_and_seed(monkeypatch):
    settings = Settings(
        api_key="test",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0.7,
        cache_enabled=False,
        cache_dir=".llm_cache",
        max_tool_iters=1,
        request_timeout=1.0,
        seed=123,
    )
    llm = LLMClient(settings=settings, cache_enabled=False)
    captured = {}

    def fake_call(messages, tools, tool_choice, force_json, *, temperature, seed):
        captured["temperature"] = temperature
        captured["seed"] = seed
        return ChatResponse(content="{}")

    monkeypatch.setattr(llm, "_call_openai", fake_call)

    llm.chat([{"role": "user", "content": "x"}], force_json=True, temperature=0.0)

    assert captured == {"temperature": 0.0, "seed": 123}


def test_cache_only_refuses_live_call_on_miss():
    llm = LLMClient(cache_only=True)
    with pytest.raises(LLMError):
        llm.chat([
            {"role": "system", "content": "unique-nonce-xyz"},
            {"role": "user", "content": "no-such-cached-request-123"},
        ])

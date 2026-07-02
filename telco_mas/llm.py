"""OpenAI-compatible LLM layer: a thin client plus a tool-calling agent loop.

Design notes
------------
* Provider-agnostic: works against OpenAI or DeepSeek (or any OpenAI-compatible
  gateway) purely through ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``.
* ``run_agent`` implements the agentic tool-use loop manually so every step is
  captured in a trace (needed for the dashboard and the report).
* An optional on-disk cache stores *real* LLM responses keyed by request hash, so
  the benchmark is cheap to re-run and produces stable numbers. It never fabricates
  reasoning — it only replays genuine completions.
* A ``responder`` callable can be injected to stub the network for unit tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import Settings, get_settings
from .schemas import ToolCall, TraceStep, UsageStats


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns malformed output."""


# A responder stubs the backend for tests: (messages, tools) -> assistant message dict.
Responder = Callable[[list[dict], Optional[list[dict]]], dict]
# A dispatcher executes a tool call: (name, arguments) -> string result.
Dispatcher = Callable[[str, dict], str]


@dataclass
class ChatResponse:
    content: Optional[str]
    tool_calls: list[dict] = field(default_factory=list)  # {id, name, arguments, arguments_raw}
    usage: UsageStats = field(default_factory=UsageStats)
    finish_reason: str = "stop"


@dataclass
class AgentRun:
    """Result of a full agent tool-use loop."""

    content: str
    data: dict[str, Any]
    trace: list[TraceStep] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)


# --------------------------------------------------------------------------- #
# JSON helpers — models are chatty; extract the object robustly.
# --------------------------------------------------------------------------- #
def extract_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # fall back to the first balanced { ... } block
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except Exception:
                    return {}
    return {}


class LLMClient:
    """Wraps an OpenAI-compatible chat backend with tracing + caching."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        responder: Optional[Responder] = None,
        cache_enabled: Optional[bool] = None,
        cache_only: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self._responder = responder
        self._client = None  # lazily created OpenAI client
        # cache_only replays cached completions and refuses to make live calls —
        # useful for reproducing a benchmark offline without spending tokens.
        self.cache_only = cache_only
        self.cache_enabled = (
            self.settings.cache_enabled if cache_enabled is None else cache_enabled
        )
        if cache_only:
            self.cache_enabled = True
        if self.cache_enabled:
            os.makedirs(self.settings.cache_dir, exist_ok=True)

    # -- backend -------------------------------------------------------------
    def _openai_client(self):
        if self._client is None:
            if not self.settings.has_api_key:
                raise LLMError(
                    "No OPENAI_API_KEY found. Set it in your environment or .env "
                    "(OpenAI) — or point OPENAI_BASE_URL at DeepSeek and use its key. "
                    "See .env.example."
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout,
            )
        return self._client

    def _cache_key(self, payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- single completion ---------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        force_json: bool = False,
    ) -> ChatResponse:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice if tools else None,
            "force_json": force_json,
            "temperature": self.settings.temperature,
        }
        cache_file = None
        if self.cache_enabled:
            cache_file = os.path.join(
                self.settings.cache_dir, self._cache_key(payload) + ".json"
            )
            if os.path.exists(cache_file):
                with open(cache_file) as fh:
                    return _response_from_dict(json.load(fh))

        if self.cache_only and self._responder is None:
            raise LLMError(
                "cache_only: no cached response for this request. The prompt/tools changed "
                "since the cache was built, so replaying it offline is not possible — re-run "
                "this configuration live (with an API key and without --cache-only)."
            )

        if self._responder is not None:
            msg = self._responder(messages, tools)
            resp = _parse_assistant_message(msg, UsageStats(llm_calls=1))
        else:
            resp = self._call_openai(messages, tools, tool_choice, force_json)

        if cache_file is not None:
            with open(cache_file, "w") as fh:
                json.dump(_response_to_dict(resp), fh)
        return resp

    def _call_openai(self, messages, tools, tool_choice, force_json) -> ChatResponse:
        client = self._openai_client()
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # pragma: no cover - network failure path
            raise LLMError(f"LLM request failed: {exc}") from exc

        choice = completion.choices[0]
        message = choice.message
        tool_calls: list[dict] = []
        for tc in message.tool_calls or []:
            raw_args = tc.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except Exception:
                parsed = {}
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": parsed, "arguments_raw": raw_args}
            )
        usage = UsageStats(llm_calls=1)
        if completion.usage:
            usage.prompt_tokens = completion.usage.prompt_tokens or 0
            usage.completion_tokens = completion.usage.completion_tokens or 0
            usage.total_tokens = completion.usage.total_tokens or 0
        return ChatResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
        )

    # -- agentic tool-use loop ----------------------------------------------
    def run_agent(
        self,
        *,
        name: str,
        system_prompt: str,
        user_prompt: str,
        tools_spec: Optional[list[dict]] = None,
        dispatcher: Optional[Dispatcher] = None,
        max_iters: Optional[int] = None,
    ) -> AgentRun:
        """Run one agent: reason, optionally call tools, return final JSON.

        The agent is instructed (via its system prompt) to finish with a JSON
        object; we parse it leniently and, if needed, issue one corrective call.
        """
        max_iters = max_iters or self.settings.max_tool_iters
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        trace: list[TraceStep] = []
        usage = UsageStats()

        for _ in range(max_iters):
            resp = self.chat(messages, tools=tools_spec, tool_choice="auto")
            usage = usage.add(resp.usage)

            if not resp.tool_calls:
                trace.append(TraceStep(agent=name, role="assistant", content=resp.content or ""))
                data = extract_json(resp.content)
                if data:
                    return AgentRun(resp.content or "", data, trace, usage)
                break  # no tools, no JSON -> corrective pass below

            # record + execute the tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments_raw"]},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )
            step_calls: list[ToolCall] = []
            for tc in resp.tool_calls:
                result = "(no dispatcher)"
                if dispatcher is not None:
                    try:
                        result = dispatcher(tc["name"], tc["arguments"])
                    except Exception as exc:  # keep the loop alive on tool errors
                        result = f"ERROR executing {tc['name']}: {exc}"
                usage.tool_calls += 1
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                step_calls.append(
                    ToolCall(
                        name=tc["name"],
                        arguments=tc["arguments"],
                        result_preview=_preview(result),
                    )
                )
            trace.append(
                TraceStep(agent=name, role="tool_use", content=resp.content or "", tool_calls=step_calls)
            )

        # corrective / finalising pass: force a JSON answer, no tools
        messages.append(
            {
                "role": "user",
                "content": "Now output ONLY the final JSON object requested in your "
                "instructions. No prose, no markdown fences.",
            }
        )
        resp = self.chat(messages, tools=None, force_json=True)
        usage = usage.add(resp.usage)
        trace.append(TraceStep(agent=name, role="assistant", content=resp.content or ""))
        return AgentRun(resp.content or "", extract_json(resp.content), trace, usage)


# --------------------------------------------------------------------------- #
# (de)serialisation helpers for caching / fake responders
# --------------------------------------------------------------------------- #
def _preview(text: str, limit: int = 280) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _parse_assistant_message(msg: dict, usage: UsageStats) -> ChatResponse:
    tool_calls = []
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", tc)
        raw = fn.get("arguments", "{}")
        parsed = raw if isinstance(raw, dict) else extract_json(raw)
        tool_calls.append(
            {
                "id": tc.get("id", "call_0"),
                "name": fn.get("name", ""),
                "arguments": parsed,
                "arguments_raw": raw if isinstance(raw, str) else json.dumps(raw),
            }
        )
    return ChatResponse(content=msg.get("content"), tool_calls=tool_calls, usage=usage)


def _response_to_dict(resp: ChatResponse) -> dict:
    return {
        "content": resp.content,
        "tool_calls": resp.tool_calls,
        "finish_reason": resp.finish_reason,
        "usage": resp.usage.model_dump(),
    }


def _response_from_dict(d: dict) -> ChatResponse:
    return ChatResponse(
        content=d.get("content"),
        tool_calls=d.get("tool_calls", []),
        usage=UsageStats(**d.get("usage", {})),
        finish_reason=d.get("finish_reason", "stop"),
    )

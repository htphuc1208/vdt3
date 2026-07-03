"""Base agent plumbing and shared prompt helpers."""
from __future__ import annotations

from typing import Any, Optional

from ..llm import AgentRun, LLMClient
from ..schemas import Incident
from ..tools.registry import SessionContext, dispatch, openai_specs


def incident_brief(incident: Incident) -> str:
    lines = [
        f"INCIDENT {incident.id}: {incident.title}",
        f"Description: {incident.description}",
        f"Affected elements (from alarms): {', '.join(incident.affected_elements) or 'unknown'}",
        "Active alarms:",
    ]
    for a in incident.alarms[:12]:
        lines.append(f"  - [{a.severity.value}] {a.element_id}: {a.name} ({a.probable_cause})")
    if len(incident.alarms) > 12:
        lines.append(f"  ... and {len(incident.alarms) - 12} more alarms")
    return "\n".join(lines)


def as_float(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
        return min(1.0, max(0.0, f))
    except Exception:
        return default


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value:
        return [str(value)]
    return []


class BaseAgent:
    """Wraps a single LLM agent with its allowed tools + tracing."""

    name: str = "agent"
    tool_names: list[str] = []

    def __init__(self, llm: LLMClient, ctx: SessionContext) -> None:
        self.llm = llm
        self.ctx = ctx

    def _dispatch(self, name: str, arguments: dict) -> str:
        return dispatch(self.ctx, name, arguments)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_names: Optional[list[str]] = None,
        max_iters: Optional[int] = None,
    ) -> AgentRun:
        names = tool_names if tool_names is not None else self.tool_names
        specs = openai_specs(names) if names else None
        return self.llm.run_agent(
            name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools_spec=specs,
            dispatcher=self._dispatch,
            max_iters=max_iters,
        )

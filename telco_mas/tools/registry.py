"""Tool registry: JSON schemas (for the LLM) + a dispatcher (to execute)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..environment.simulator import NetworkSimulator
from ..knowledge.retriever import TfidfRetriever, build_default_retriever
from . import action_tools, kb_tools, telemetry_tools


@dataclass
class SessionContext:
    """Everything the tools need to observe/act during one incident."""

    sim: NetworkSimulator
    retriever: TfidfRetriever

    @classmethod
    def create(cls, sim: NetworkSimulator, retriever: TfidfRetriever | None = None) -> "SessionContext":
        return cls(sim=sim, retriever=retriever or build_default_retriever())


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]


_S = {"type": "string"}


TOOLS: dict[str, ToolDef] = {
    "query_topology": ToolDef(
        "query_topology",
        "Inspect the network topology. With no args, lists all elements. With element_id, "
        "returns that element plus its upstream parents and downstream descendants — use this "
        "to reason about dependency and blast radius.",
        {
            "type": "object",
            "properties": {
                "element_id": {"type": "string", "description": "Element id, e.g. RAN-CELL-03"},
                "domain": {"type": "string", "enum": ["RAN", "TRANSPORT", "CORE", "POWER"]},
            },
        },
        telemetry_tools.query_topology,
    ),
    "query_alarms": ToolDef(
        "query_alarms",
        "List active alarms, optionally filtered by element_id or severity (CRITICAL/MAJOR/MINOR/WARNING).",
        {
            "type": "object",
            "properties": {
                "element_id": _S,
                "severity": {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR", "WARNING"]},
            },
        },
        telemetry_tools.query_alarms,
    ),
    "query_kpis": ToolDef(
        "query_kpis",
        "Read KPI/performance counters. With element_id, returns all its KPIs. With no element_id, "
        "returns only anomalous KPIs across the whole network. Optional metric filter.",
        {
            "type": "object",
            "properties": {"element_id": _S, "metric": _S},
        },
        telemetry_tools.query_kpis,
    ),
    "query_logs": ToolDef(
        "query_logs",
        "Read recent log lines, optionally filtered by element_id or level.",
        {
            "type": "object",
            "properties": {"element_id": _S, "level": _S, "limit": {"type": "integer"}},
        },
        telemetry_tools.query_logs,
    ),
    "search_knowledge_base": ToolDef(
        "search_knowledge_base",
        "Retrieve the most relevant Standard Operating Procedures (SOP playbooks) for a symptom/query.",
        {
            "type": "object",
            "properties": {"query": _S, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        kb_tools.search_knowledge_base,
    ),
    "get_historical_incidents": ToolDef(
        "get_historical_incidents",
        "Retrieve similar past incidents (symptoms, root cause, resolution) for a query.",
        {
            "type": "object",
            "properties": {"query": _S, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        kb_tools.get_historical_incidents,
    ),
    "run_diagnostic": ToolDef(
        "run_diagnostic",
        "Run active diagnostics on an element (e.g. interface_status, optical_power, cell_status, "
        "config_audit, hardware, power, resource, dns, license, rf). Omit 'check' to run all relevant checks.",
        {
            "type": "object",
            "properties": {"element_id": _S, "check": _S},
            "required": ["element_id"],
        },
        action_tools.run_diagnostic,
    ),
    "apply_remediation": ToolDef(
        "apply_remediation",
        "Apply a remediation action to the network (simulated). Provide a clear 'action' describing the "
        "fix and the target 'element_id'. Returns whether KPIs recovered — only a correct fix on the "
        "correct element resolves the incident.",
        {
            "type": "object",
            "properties": {"action": _S, "element_id": _S},
            "required": ["action"],
        },
        action_tools.apply_remediation,
    ),
}


def openai_specs(names: list[str] | None = None) -> list[dict]:
    selected = names or list(TOOLS)
    return [
        {
            "type": "function",
            "function": {
                "name": TOOLS[n].name,
                "description": TOOLS[n].description,
                "parameters": TOOLS[n].parameters,
            },
        }
        for n in selected
        if n in TOOLS
    ]


def dispatch(ctx: SessionContext, name: str, arguments: dict[str, Any]) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'"
    return tool.fn(ctx, **(arguments or {}))

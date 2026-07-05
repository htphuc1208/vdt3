"""Budgeted local telemetry tools for single-agent ShardRCA baselines."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .board import Blackboard, Finding
from .catalog import ShardSpec, TelemetryCatalog, make_component_group_shards, make_default_shards
from .mining import mine_shard


LOCAL_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_evidence_shards",
            "description": "List available label-safe telemetry shards. Does not return evidence.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_evidence_shard",
            "description": "Mine one local telemetry shard and return compact evidence for that shard only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shard_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["shard_id"],
                "additionalProperties": False,
            },
        },
    },
]


@dataclass
class LocalToolSession:
    catalog: TelemetryCatalog
    chunksize: int = 50_000
    finding_limit: int = 10
    max_tool_calls: int = 3
    shards: list[ShardSpec] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    calls_used: int = 0

    def __post_init__(self) -> None:
        if not self.shards:
            self.shards = (
                make_component_group_shards(self.catalog, group_size=6)
                if str(self.catalog.dataset).startswith("RE")
                else make_default_shards(self.catalog, max_shards=6, prefer_simple_metrics=False)
            )
        component_filter = [
            str(item)
            for item in self.catalog.metadata.get("candidate_components", [])
        ]
        if component_filter:
            self.shards = [
                shard.model_copy(update={"components": component_filter})
                for shard in self.shards
            ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        if self.calls_used >= self.max_tool_calls:
            return json.dumps({"error": "tool budget exhausted", "max_tool_calls": self.max_tool_calls})
        self.calls_used += 1
        if name == "list_evidence_shards":
            return json.dumps(self.list_shards(), ensure_ascii=True)
        if name == "query_evidence_shard":
            shard_id = str(arguments.get("shard_id") or "")
            limit = _bounded_limit(arguments.get("limit"), self.finding_limit)
            return json.dumps(self.query_shard(shard_id, limit=limit), ensure_ascii=True, default=str)
        return json.dumps({"error": f"unknown tool {name}"})

    def list_shards(self) -> dict[str, Any]:
        return {
            "case_id": self.catalog.case_id,
            "max_tool_calls": self.max_tool_calls,
            "calls_used": self.calls_used,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "modality": shard.modality,
                    "file_count": len(shard.paths),
                    "bytes": shard.metadata.get("bytes", 0),
                    "components": shard.components,
                    "hint": "call query_evidence_shard on this shard to inspect evidence",
                }
                for shard in self.shards
            ],
        }

    def query_shard(self, shard_id: str, *, limit: int) -> dict[str, Any]:
        shard = next((item for item in self.shards if item.shard_id == shard_id), None)
        if shard is None:
            return {"error": f"unknown shard_id {shard_id}", "available": [item.shard_id for item in self.shards]}
        findings = mine_shard(shard, limit=limit, chunksize=self.chunksize)
        self.findings.extend(findings)
        return {
            "shard_id": shard.shard_id,
            "modality": shard.modality,
            "findings": [finding.compact() for finding in findings],
            "calls_used": self.calls_used,
            "remaining_calls": max(0, self.max_tool_calls - self.calls_used),
        }

    def board(self) -> Blackboard:
        board = Blackboard(case_id=self.catalog.case_id, catalog_summary=self.catalog.summary())
        board.extend(self.findings)
        return board


def heuristic_probe(session: LocalToolSession) -> Blackboard:
    """Offline fallback: inspect shards in a fixed budgeted order."""

    session.dispatch("list_evidence_shards", {})
    priority = {"metrics": 0, "traces": 1, "logs": 2}
    for shard in sorted(session.shards, key=lambda item: priority.get(item.modality, 9)):
        if session.calls_used >= session.max_tool_calls:
            break
        session.dispatch("query_evidence_shard", {"shard_id": shard.shard_id, "limit": session.finding_limit})
    return session.board()


def _bounded_limit(value: Any, default: int) -> int:
    try:
        limit = int(value)
    except Exception:
        limit = default
    return max(1, min(limit, 20))

"""Typed blackboard for ShardRCA findings and candidate answers."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["metrics", "logs", "traces", "events", "auxiliary"]


class Finding(BaseModel):
    """A compact, label-safe observation produced by one shard miner."""

    shard_id: str
    modality: Modality
    component: str
    signal: str
    direction: str = ""
    magnitude: float = 0.0
    score: float = 0.0
    window_start: str | float | int | None = None
    window_end: str | float | int | None = None
    evidence_ptr: str = ""
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.modality,
            self.component.strip().lower(),
            self.signal.strip().lower(),
            self.direction.strip().lower(),
        )

    def compact(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "modality": self.modality,
            "signal": self.signal,
            "direction": self.direction,
            "score": round(float(self.score), 4),
            "magnitude": round(float(self.magnitude), 4),
            "window": [self.window_start, self.window_end],
            "evidence": self.evidence_ptr,
            "summary": self.summary,
        }


class CandidateRootCause(BaseModel):
    component: str
    reason: str = "unknown"
    occurrence_time: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    score: float = 0.0

    def compact(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "reason": self.reason,
            "occurrence_time": self.occurrence_time,
            "confidence": round(float(self.confidence), 4),
            "score": round(float(self.score), 4),
            "evidence": self.evidence[:6],
            "rationale": self.rationale,
        }


class CandidateEvidence(BaseModel):
    """Local candidate emitted by one evidence-isolated shard worker."""

    component: str
    reason_family: str = "unknown"
    support_score: float = 0.0
    refute_score: float = 0.0
    probability: float = Field(0.0, ge=0.0, le=1.0)
    modality: Modality = "auxiliary"
    worker_id: str = ""
    shard_id: str = ""
    evidence_ptrs: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    local_rank: int = 1
    rationale: str = ""

    def compact(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "reason_family": self.reason_family,
            "support_score": round(float(self.support_score), 4),
            "refute_score": round(float(self.refute_score), 4),
            "probability": round(float(self.probability), 6),
            "modality": self.modality,
            "worker_id": self.worker_id,
            "shard_id": self.shard_id,
            "local_rank": self.local_rank,
            "evidence": self.evidence_ptrs[:6],
            "missing_evidence": self.missing_evidence[:6],
            "rationale": self.rationale,
        }


class WorkerDistribution(BaseModel):
    """A normalized local posterior emitted by one evidence-isolated worker."""

    worker_id: str
    modality: Modality
    candidate_scope: list[str]
    candidates: list[CandidateEvidence] = Field(default_factory=list)
    other_mass: float = Field(0.0, ge=0.0, le=1.0)
    notes: str = ""

    def compact(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "modality": self.modality,
            "candidate_scope": self.candidate_scope,
            "candidates": [candidate.compact() for candidate in self.candidates],
            "other_mass": round(float(self.other_mass), 6),
            "notes": self.notes,
        }


class Blackboard(BaseModel):
    """Shared evidence board consumed by synthesizers and falsifiers."""

    case_id: str
    findings: list[Finding] = Field(default_factory=list)
    catalog_summary: dict[str, Any] = Field(default_factory=dict)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def ranked_findings(self, limit: int | None = None) -> list[Finding]:
        ranked = dedup_findings(self.findings)
        if limit is not None:
            return ranked[: max(0, limit)]
        return ranked

    def component_scores(self, *, top_per_modality: int = 3) -> dict[str, float]:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for finding in self.ranked_findings():
            if finding.component:
                grouped[finding.component][finding.modality].append(max(0.0, float(finding.score)))
        scores = {
            component: sum(
                sum(sorted(modality_scores, reverse=True)[:top_per_modality])
                for modality_scores in modalities.values()
            )
            for component, modalities in grouped.items()
        }
        return {key: round(value, 6) for key, value in scores.items()}

    def component_score(self, component: str, *, top_per_modality: int = 3) -> float:
        needle = component.strip().lower()
        return next(
            (score for name, score in self.component_scores(top_per_modality=top_per_modality).items() if name.strip().lower() == needle),
            0.0,
        )

    def top_components(self, limit: int = 10) -> list[tuple[str, float]]:
        return sorted(self.component_scores().items(), key=lambda item: item[1], reverse=True)[:limit]

    def evidence_for(self, component: str, limit: int = 8) -> list[Finding]:
        needle = component.strip().lower()
        return [
            finding
            for finding in self.ranked_findings()
            if finding.component.strip().lower() == needle
        ][:limit]

    def compact_render(self, max_findings: int = 40) -> str:
        payload = {
            "case_id": self.case_id,
            "catalog": self.catalog_summary,
            "top_components": self.top_components(12),
            "findings": [finding.compact() for finding in self.ranked_findings(max_findings)],
        }
        return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


def dedup_findings(findings: list[Finding]) -> list[Finding]:
    """Keep the strongest evidence item for each semantic finding key."""

    by_key: dict[tuple[str, str, str, str], Finding] = {}
    for finding in findings:
        current = by_key.get(finding.key)
        if current is None or finding.score > current.score:
            by_key[finding.key] = finding
    return sorted(
        by_key.values(),
        key=lambda item: (float(item.score), float(item.magnitude)),
        reverse=True,
    )

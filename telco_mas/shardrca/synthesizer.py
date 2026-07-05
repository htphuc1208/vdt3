"""Candidate synthesis and voting for ShardRCA."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..agents.consensus import tally_votes
from ..llm import LLMClient, extract_json
from ..schemas import Hypothesis, UsageStats
from .board import Blackboard, CandidateRootCause, Finding


SYNTHESIZER_SYSTEM = """You are a ShardRCA synthesizer.
Use only the label-safe blackboard findings. Pick the most likely root-cause component,
fault/reason family, and occurrence time if the evidence supports one.
Return ONLY JSON:
{"component": "service_or_component", "reason": "cpu|mem|disk|delay|loss|socket|network|db|unknown",
 "occurrence_time": "optional timestamp or null", "confidence": 0.0,
 "rationale": "short evidence-based explanation", "evidence": ["evidence_ptr", "..."]}"""


@dataclass
class SynthesizerResult:
    winner: CandidateRootCause
    candidates: list[CandidateRootCause]
    vote_breakdown: dict[str, float] = field(default_factory=dict)
    usage: UsageStats = field(default_factory=UsageStats)


def synthesize(
    board: Blackboard,
    *,
    llm: LLMClient | None = None,
    k: int = 3,
    max_findings: int = 40,
) -> SynthesizerResult:
    usage = UsageStats()
    candidates: list[CandidateRootCause] = []
    if llm is not None:
        prompt = board.compact_render(max_findings=max_findings)
        for _ in range(max(1, k)):
            response = llm.chat(
                [{"role": "system", "content": SYNTHESIZER_SYSTEM}, {"role": "user", "content": prompt}],
                force_json=True,
            )
            usage = usage.add(response.usage)
            candidate = _candidate_from_json(extract_json(response.content), board)
            if candidate.component and candidate.component != "UNKNOWN":
                candidates.append(candidate)

    if not candidates:
        candidates = heuristic_candidates(board, k=max(1, k))

    winner, vote_breakdown = fuse_candidates(candidates)
    return SynthesizerResult(winner=winner, candidates=candidates, vote_breakdown=vote_breakdown, usage=usage)


def heuristic_candidates(board: Blackboard, *, k: int = 3) -> list[CandidateRootCause]:
    top = board.top_components(max(k, 1))
    if not top:
        return [
            CandidateRootCause(
                component="UNKNOWN",
                reason="unknown",
                confidence=0.0,
                rationale="No telemetry findings were available.",
            )
        ]
    total = sum(score for _, score in top) or 1.0
    candidates: list[CandidateRootCause] = []
    for component, score in top[: max(1, k)]:
        evidence = board.evidence_for(component, limit=8)
        reason = infer_reason(evidence)
        candidates.append(
            CandidateRootCause(
                component=component,
                reason=reason,
                occurrence_time=_first_occurrence(evidence),
                confidence=max(0.05, min(0.95, score / total)),
                score=score,
                evidence=[item.evidence_ptr for item in evidence if item.evidence_ptr][:8],
                rationale=_rationale(component, reason, evidence),
            )
        )
    return candidates


def fuse_candidates(candidates: list[CandidateRootCause]) -> tuple[CandidateRootCause, dict[str, float]]:
    if not candidates:
        return CandidateRootCause(component="UNKNOWN", reason="unknown"), {}
    hypotheses = [
        Hypothesis(
            proposed_by=f"synth_{idx}",
            root_cause=candidate.rationale or f"{candidate.reason} on {candidate.component}",
            faulty_element_id=candidate.component,
            fault_type=candidate.reason,
            confidence=candidate.confidence,
            evidence=list(candidate.evidence),
        )
        for idx, candidate in enumerate(candidates)
    ]
    scores, _ = tally_votes(hypotheses, calibration={})
    top_component = max(scores, key=scores.get) if scores else candidates[0].component
    backers = [candidate for candidate in candidates if candidate.component == top_component]
    chosen = max(backers or candidates, key=lambda item: (item.confidence, item.score))
    total = sum(scores.values()) or 1.0
    return (
        CandidateRootCause(
            component=chosen.component,
            reason=chosen.reason,
            occurrence_time=chosen.occurrence_time,
            confidence=round(scores.get(top_component, chosen.confidence) / total, 4),
            rationale=chosen.rationale,
            evidence=chosen.evidence,
            score=max(item.score for item in backers) if backers else chosen.score,
        ),
        {key: round(value, 4) for key, value in scores.items()},
    )


def infer_reason(findings: list[Finding]) -> str:
    text = " ".join(f"{item.signal} {item.summary} {item.direction}" for item in findings).lower()
    for reason, needles in {
        "cpu": ("cpu", "processor"),
        "mem": ("mem", "memory", "rss", "heap"),
        "disk": ("disk", "fs-", "blkio", "io"),
        "loss": ("loss", "drop", "dropped", "error-total", "errors"),
        "delay": ("delay", "latency", "duration", "timeout", "slow"),
        "socket": ("socket", "sockets", "connection reset"),
        "db": ("db", "database", "sql", "connection limit", "close"),
        "network": ("network", "rx", "tx", "packet"),
    }.items():
        if any(needle in text for needle in needles):
            return reason
    return "unknown"


def _candidate_from_json(data: dict[str, Any], board: Blackboard) -> CandidateRootCause:
    component = str(data.get("component") or data.get("root") or data.get("root_cause_component") or "").strip()
    if not component:
        component = board.top_components(1)[0][0] if board.top_components(1) else "UNKNOWN"
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    board_evidence = board.evidence_for(component, limit=8)
    return CandidateRootCause(
        component=component,
        reason=str(data.get("reason") or infer_reason(board_evidence) or "unknown"),
        occurrence_time=_optional_str(data.get("occurrence_time")) or _first_occurrence(board_evidence),
        confidence=max(0.0, min(1.0, confidence)),
        rationale=str(data.get("rationale") or _rationale(component, "unknown", board_evidence)),
        evidence=[str(item) for item in evidence] or [item.evidence_ptr for item in board_evidence if item.evidence_ptr],
        score=board.component_score(component),
    )


def _optional_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _first_occurrence(findings: list[Finding]) -> str | None:
    for finding in findings:
        if finding.window_start is not None:
            return str(finding.window_start)
    return None


def _rationale(component: str, reason: str, findings: list[Finding]) -> str:
    if not findings:
        return f"{component} was selected by aggregate shard evidence; reason={reason}."
    snippets = [
        f"{item.modality}:{item.signal} {item.direction} score={item.score:.2f}"
        for item in findings[:4]
    ]
    return f"{component} has the strongest shard evidence for {reason}: " + "; ".join(snippets)


def candidates_json(candidates: list[CandidateRootCause]) -> str:
    return json.dumps([candidate.compact() for candidate in candidates], indent=2, ensure_ascii=True)

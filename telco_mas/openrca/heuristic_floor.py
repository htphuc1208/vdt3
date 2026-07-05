"""No-LLM OpenRCA heuristic floor over prepared telemetry."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..schemas import UsageStats
from ..shardrca.board import Blackboard, CandidateRootCause, Finding
from ..shardrca.synthesizer import SynthesizerResult
from .prepared import PreparedOpenRCA
from .tools import TELECOM_REASONS, candidate_catalog_for_row
from .workers import run_isolated_workers


@dataclass
class HeuristicFloorResult:
    board: Blackboard
    synthesis: SynthesizerResult
    winner: CandidateRootCause
    usage: UsageStats
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: str = "no-LLM prepared-telemetry heuristic floor"


def run_heuristic_floor(
    prepared: PreparedOpenRCA,
    row_id: int,
    *,
    candidate_catalog: dict[str, Any] | None = None,
    finding_limit: int = 30,
) -> HeuristicFloorResult:
    """Return a deterministic floor prediction from metric/trace anomaly evidence."""

    catalog = candidate_catalog or candidate_catalog_for_row(prepared, row_id)
    workers = run_isolated_workers(
        prepared,
        row_id,
        llm=None,
        finding_limit=finding_limit,
        candidate_components=list(catalog["components"]),
        candidate_reasons=list(catalog["reasons"]),
        candidate_catalog_source=dict(catalog["source"]),
    )
    candidates = _candidates_from_board(workers.board, list(catalog["components"]), list(catalog["reasons"]))
    winner = candidates[0] if candidates else CandidateRootCause(component="UNKNOWN", reason="unknown")
    synthesis = SynthesizerResult(
        winner=winner,
        candidates=candidates or [winner],
        vote_breakdown={candidate.component: round(candidate.score, 6) for candidate in candidates},
    )
    return HeuristicFloorResult(
        board=workers.board,
        synthesis=synthesis,
        winner=winner,
        usage=workers.usage,
        artifacts={
            "architecture": "prepared_telemetry_heuristic_floor",
            "candidate_catalog_source": catalog["source"],
            "candidate_catalog": {
                "component_count": len(catalog["components"]),
                "reason_count": len(catalog["reasons"]),
                "components": list(catalog["components"]),
                "reasons": list(catalog["reasons"]),
            },
            "worker_diagnostics": workers.diagnostics,
            "top_components": workers.board.top_components(8),
        },
    )


def _candidates_from_board(
    board: Blackboard,
    components: list[str],
    reasons: list[str],
    *,
    limit: int = 8,
) -> list[CandidateRootCause]:
    top = [
        (component, score)
        for component, score in board.top_components(limit)
        if component in set(components)
    ]
    if not top and components:
        top = [(components[0], 0.0)]
    total = sum(max(0.0, score) for _, score in top) or 1.0
    out: list[CandidateRootCause] = []
    for component, score in top:
        evidence = board.evidence_for(component, limit=8)
        reason = _reason_from_evidence(evidence, reasons)
        out.append(
            CandidateRootCause(
                component=component,
                reason=reason,
                occurrence_time=_first_occurrence(evidence),
                confidence=max(0.0, min(0.95, max(0.0, score) / total)),
                rationale=(
                    "No-LLM floor selected the strongest prepared-telemetry "
                    f"anomaly component with reason_hint={reason}."
                ),
                evidence=[finding.evidence_ptr for finding in evidence if finding.evidence_ptr][:6],
                score=float(score),
            )
        )
    return out


def _reason_from_evidence(evidence: list[Finding], reasons: list[str]) -> str:
    allowed = set(reasons or TELECOM_REASONS)
    weighted: Counter[str] = Counter()
    for finding in evidence:
        hint = str(finding.metadata.get("reason_hint") or "")
        if hint in allowed:
            weighted[hint] += max(0.0, float(finding.score))
    if weighted:
        return weighted.most_common(1)[0][0]
    return next(iter(reasons or TELECOM_REASONS), "unknown")


def _first_occurrence(evidence: list[Finding]) -> str | None:
    for finding in evidence:
        if finding.window_start is not None:
            return str(finding.window_start)
    return None

"""Targeted evidence verification pass for ShardRCA candidates.

The public function keeps the historical ``falsify`` name for API
compatibility, but this is a threshold-free evidence verifier/reranker rather
than a Popperian adversarial falsification procedure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import UsageStats
from .board import Blackboard, CandidateRootCause


@dataclass
class FalsifierResult:
    winner: CandidateRootCause
    falsified: bool
    notes: str
    checks: list[str] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)


def falsify(
    board: Blackboard,
    candidates: list[CandidateRootCause],
    *,
    top: CandidateRootCause | None = None,
    min_evidence_score: float | None = None,
    runner_up_margin: float | None = None,
) -> FalsifierResult:
    if not candidates:
        return FalsifierResult(
            winner=CandidateRootCause(component="UNKNOWN", reason="unknown"),
            falsified=False,
            notes="No candidates to verify.",
        )
    top = top or candidates[0]
    ranked = sorted(candidates, key=lambda item: (item.score, item.confidence), reverse=True)
    top_score = _board_score(board, top.component)
    checks = [
        "decision_rule=promote_runner_if_board_score_strictly_higher",
        f"top_component={top.component}",
        f"top_board_score={top_score:.4f}",
        f"top_candidate_confidence={top.confidence:.4f}",
    ]
    if min_evidence_score is not None or runner_up_margin is not None:
        checks.append("deprecated_threshold_args_ignored=true")
    runner = next((item for item in ranked if item.component != top.component), None)
    if runner is not None:
        runner_score = _board_score(board, runner.component)
        checks.append(f"runner_up={runner.component}")
        checks.append(f"runner_up_board_score={runner_score:.4f}")
        if runner_score > top_score:
            return FalsifierResult(
                winner=runner,
                falsified=True,
                notes=(
                    f"Runner-up {runner.component} has higher targeted board evidence "
                    f"than {top.component}; promoted runner-up."
                ),
                checks=checks,
                usage=UsageStats(tool_calls=1),
            )
    return FalsifierResult(
        winner=top,
        falsified=False,
        notes=f"Targeted evidence verifier kept {top.component}; no runner-up had higher board evidence.",
        checks=checks,
        usage=UsageStats(tool_calls=1),
    )


def _board_score(board: Blackboard, component: str) -> float:
    return board.component_score(component)

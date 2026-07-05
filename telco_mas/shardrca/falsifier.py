"""Adversarial verification pass for ShardRCA candidates."""
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
    min_evidence_score: float = 0.1,
    runner_up_margin: float = 1.25,
) -> FalsifierResult:
    if not candidates:
        return FalsifierResult(
            winner=CandidateRootCause(component="UNKNOWN", reason="unknown"),
            falsified=False,
            notes="No candidates to falsify.",
        )
    top = top or candidates[0]
    ranked = sorted(candidates, key=lambda item: (item.score, item.confidence), reverse=True)
    top_score = _board_score(board, top.component)
    checks = [
        f"top_component={top.component}",
        f"top_board_score={top_score:.4f}",
        f"top_candidate_confidence={top.confidence:.4f}",
    ]
    if top_score < min_evidence_score:
        runner = ranked[0] if ranked and ranked[0].component != top.component else (ranked[1] if len(ranked) > 1 else top)
        if runner.component != top.component:
            return FalsifierResult(
                winner=runner,
                falsified=True,
                notes=f"Top candidate {top.component} lacked targeted board evidence; promoted {runner.component}.",
                checks=checks + [f"promoted={runner.component}"],
                usage=UsageStats(tool_calls=1),
            )
    runner = next((item for item in ranked if item.component != top.component), None)
    if runner is not None:
        runner_score = _board_score(board, runner.component)
        checks.append(f"runner_up={runner.component}")
        checks.append(f"runner_up_board_score={runner_score:.4f}")
        if runner_score > max(top_score * runner_up_margin, top_score + min_evidence_score):
            return FalsifierResult(
                winner=runner,
                falsified=True,
                notes=(
                    f"Runner-up {runner.component} had stronger targeted evidence "
                    f"than {top.component}; promoted runner-up."
                ),
                checks=checks,
                usage=UsageStats(tool_calls=1),
            )
    return FalsifierResult(
        winner=top,
        falsified=False,
        notes=f"Targeted evidence supports {top.component}; no promotion.",
        checks=checks,
        usage=UsageStats(tool_calls=1),
    )


def _board_score(board: Blackboard, component: str) -> float:
    return board.component_score(component)

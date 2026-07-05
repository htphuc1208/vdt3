"""Temporal-precedence re-ranking for ShardRCA fusion.

Classic RCA prior: a root fault's anomaly *precedes* the downstream symptoms it
causes. Each board finding already carries an onset (``Finding.window_start``), so
we can compute the earliest onset per component and boost early-onset candidates
over late-only ones (which are more likely propagated symptoms).

Gated by ``FusionWeights.temporal_beta`` (default 0.0 = no-op). Like the topology
re-rank, this can change the winner (it is not a monotone rescale of one score).
Robust to coarse/ties: components sharing the earliest onset all receive full
precedence, so the boost only *breaks* ties in favour of components that a wrong
top candidate does not share (i.e. it demotes strictly-later symptom components).
"""
from __future__ import annotations

from collections import defaultdict

from .board import Blackboard, CandidateRootCause
from .synthesizer import SynthesizerResult


def component_onsets(board: Blackboard) -> dict[str, float]:
    """Earliest numeric onset (window_start) per component, lower-cased key."""
    onsets: dict[str, list[float]] = defaultdict(list)
    for finding in board.findings:
        if finding.window_start is None or not finding.component:
            continue
        try:
            onsets[finding.component.strip().lower()].append(float(finding.window_start))
        except (TypeError, ValueError):
            continue
    return {component: min(values) for component, values in onsets.items() if values}


def precedence_scores(onsets: dict[str, float]) -> dict[str, float]:
    """Map onsets to precedence in [0, 1]; 1 = earliest, 0 = latest. Ties keep 1."""
    if not onsets:
        return {}
    earliest = min(onsets.values())
    latest = max(onsets.values())
    span = latest - earliest
    if span <= 0:
        return {component: 1.0 for component in onsets}
    return {component: 1.0 - (onset - earliest) / span for component, onset in onsets.items()}


def temporal_rerank(
    result: SynthesizerResult,
    board: Blackboard,
    *,
    beta: float,
    max_candidates: int = 8,
) -> SynthesizerResult:
    """Re-rank fused candidates by ``score * (1 + beta * precedence)``.

    ``beta == 0`` or missing onset data returns the input unchanged.
    """
    if beta <= 0.0 or not result.candidates:
        return result
    precedence = precedence_scores(component_onsets(board))
    if not precedence:
        return result

    rescored: list[tuple[CandidateRootCause, float]] = []
    for cand in result.candidates:
        p = precedence.get(cand.component.strip().lower(), 0.0)
        rescored.append((cand, float(cand.score) * (1.0 + beta * p)))
    rescored.sort(key=lambda item: item[1], reverse=True)

    total = sum(score for _, score in rescored) or 1.0
    candidates: list[CandidateRootCause] = []
    for cand, score in rescored[:max_candidates]:
        candidates.append(cand.model_copy(update={
            "score": round(score, 6),
            "confidence": max(0.05, min(0.95, score / total)),
        }))
    return SynthesizerResult(
        winner=candidates[0],
        candidates=candidates,
        vote_breakdown={c.component: round(c.score, 4) for c in candidates},
        usage=result.usage,
    )

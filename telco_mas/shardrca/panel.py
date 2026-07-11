"""Grounded final adjudication for the ShardRCA investigator panel.

The panel keeps the useful bounded shard investigations, but avoids treating a
mechanical sum of anomaly strength as causal judgment.  It exposes the final
adjudicator to three independently produced views:

* local shard candidates (including peer-review revisions),
* a holistic reading of the merged, label-safe blackboard, and
* a deterministic pre/post metric-shift specialist.

Only candidates already proposed by one of those views are eligible.  Hidden
labels, engineering case paths, and case-name-derived fault hints never enter
the prompt.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .board import Blackboard, CandidateRootCause
from .synthesizer import SynthesizerResult
from .topology import DependencyGraph


PANEL_SYSTEM = """You are the final adjudicator for an RCA investigator panel.
All inputs are label-safe proposals derived from telemetry. Choose the most
likely ROOT-CAUSE component, not the loudest propagated symptom.

Decision rules:
- You may choose ONLY a component in eligible_components.
- Treat local fusion, the holistic investigator, peer reviews, and metric shifts
  as fallible specialists. Agreement is useful but is not proof.
- A large pre/post metric shift can be either a root signal or a downstream
  symptom. Prefer it when its lead over the runner-up is strong and the board
  contains direct evidence for that component.
- When A calls B and both are anomalous, prefer B only when B's failure explains
  A; do not blindly prefer hubs, databases, or leaves.
- Challenges without direct same-component evidence are abstentions.
- Base the answer on supplied evidence; never infer from case IDs or names.

Return ONLY JSON:
{"root":"eligible component", "reason":"cpu|mem|disk|delay|loss|socket|network|db|unknown",
 "confidence":0.0, "ranked_roots":["c1","c2","c3"],
 "rationale":"short evidence-based adjudication"}"""


def metric_specialist(task_payload: dict[str, Any] | None, *, limit: int = 8) -> dict[str, Any]:
    """Build a finite, label-safe metric-shift opinion from inference payload."""

    observability = (task_payload or {}).get("observability")
    raw = observability.get("top_metric_shifts", []) if isinstance(observability, dict) else []
    shifts: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        component = str(item.get("service") or "").strip()
        try:
            score = float(item.get("score"))
        except Exception:
            continue
        if not component or not math.isfinite(score):
            continue
        shifts.append({
            "component": component,
            "metric": str(item.get("metric") or "unknown"),
            "score": round(score, 6),
        })
    shifts.sort(key=lambda item: float(item["score"]), reverse=True)
    shifts = _dedupe_metric_components(shifts)[: max(0, limit)]
    top = float(shifts[0]["score"]) if shifts else 0.0
    runner = float(shifts[1]["score"]) if len(shifts) > 1 else 0.0
    margin = top / (top + runner + 1e-9) if top > 0 and runner > 0 else (1.0 if top > 0 else 0.0)
    return {
        "proposal": shifts[0]["component"] if shifts else None,
        "confidence_margin": round(margin, 4),
        "ranked_shifts": shifts,
    }


def adjudicate_panel(
    mechanical: SynthesizerResult,
    holistic: SynthesizerResult,
    board: Blackboard,
    *,
    interaction_transcript: list[dict[str, Any]] | None,
    task_payload: dict[str, Any] | None,
    graph: DependencyGraph | None,
    llm: LLMClient | None,
    k: int = 3,
    max_candidates: int = 10,
) -> tuple[SynthesizerResult, dict[str, Any]]:
    """Adjudicate specialist proposals and return a grounded final ranking.

    With no LLM, the holistic result is retained so deterministic smoke tests do
    not pretend that a heuristic is a learned adjudicator.
    """

    metric = metric_specialist(task_payload)
    eligible = _eligible_candidates(mechanical, holistic, metric, max_candidates=max_candidates)
    diagnostic: dict[str, Any] = {
        "enabled": llm is not None and len(eligible) >= 2,
        "eligible_components": eligible,
        "mechanical_proposal": mechanical.winner.component,
        "holistic_proposal": holistic.winner.component,
        "metric_specialist": metric,
        "valid_votes": 0,
    }
    if llm is None or len(eligible) < 2:
        diagnostic["fallback"] = "holistic"
        return holistic, diagnostic

    payload = {
        "eligible_components": eligible,
        "specialists": {
            "local_peer_fusion": _compact_candidates(mechanical, max_candidates),
            "holistic_board_reader": _compact_candidates(holistic, max_candidates),
            "metric_shift_specialist": metric,
        },
        "candidate_evidence": {
            component: [finding.compact() for finding in board.evidence_for(component, limit=5)]
            for component in eligible
        },
        "peer_reviews": _compact_peer_reviews(interaction_transcript or [], eligible),
        "dependency_graph": _graph_context(graph, eligible),
    }
    allowed = {component.strip().lower(): component for component in eligible}
    votes: dict[str, float] = defaultdict(float)
    rank_points: dict[str, float] = defaultdict(float)
    best: dict[str, CandidateRootCause] = {}
    usage = UsageStats()
    for _ in range(max(1, k)):
        response = llm.chat(
            [
                {"role": "system", "content": PANEL_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
            ],
            force_json=True,
            temperature=0.0,
        )
        usage = usage.add(response.usage)
        data = extract_json(response.content)
        key = str(data.get("root") or "").strip().lower()
        if key not in allowed:
            continue
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
        except Exception:
            confidence = 0.0
        votes[key] += 1.0 + confidence
        ranked = data.get("ranked_roots") if isinstance(data.get("ranked_roots"), list) else []
        for rank, component in enumerate(ranked[: len(eligible)]):
            ranked_key = str(component).strip().lower()
            if ranked_key in allowed:
                rank_points[ranked_key] += len(eligible) - rank
        base = _candidate_for_component(key, mechanical, holistic, board)
        best[key] = base.model_copy(update={
            "reason": str(data.get("reason") or base.reason or "unknown"),
            "confidence": confidence,
            "rationale": str(data.get("rationale") or base.rationale),
        })

    diagnostic["valid_votes"] = len([value for value in votes.values() if value > 0])
    diagnostic["vote_breakdown"] = {allowed[key]: round(value, 4) for key, value in votes.items()}
    if not votes:
        diagnostic["fallback"] = "holistic_invalid_panel_output"
        return holistic, diagnostic

    winner_key = max(votes, key=lambda key: (votes[key], rank_points[key]))
    ordered_keys = sorted(
        allowed,
        key=lambda key: (
            key == winner_key,
            votes.get(key, 0.0),
            rank_points.get(key, 0.0),
            _candidate_for_component(key, mechanical, holistic, board).score,
        ),
        reverse=True,
    )
    candidates = [best.get(key) or _candidate_for_component(key, mechanical, holistic, board) for key in ordered_keys]
    winner = best.get(winner_key) or candidates[0]
    candidates = [winner, *[item for item in candidates if item.component.strip().lower() != winner_key]]
    return (
        SynthesizerResult(
            winner=winner,
            candidates=candidates,
            vote_breakdown={allowed[key]: round(value, 4) for key, value in votes.items()},
            usage=usage,
        ),
        diagnostic,
    )


def _eligible_candidates(
    mechanical: SynthesizerResult,
    holistic: SynthesizerResult,
    metric: dict[str, Any],
    *,
    max_candidates: int,
) -> list[str]:
    values = [
        holistic.winner.component,
        mechanical.winner.component,
        *[item.component for item in holistic.candidates],
        *[item.component for item in mechanical.candidates],
        *[str(item.get("component") or "") for item in metric.get("ranked_shifts", [])[:4]],
    ]
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in {"unknown"} and key not in seen:
            out.append(text)
            seen.add(key)
        if len(out) >= max_candidates:
            break
    return out


def _candidate_for_component(
    key: str,
    mechanical: SynthesizerResult,
    holistic: SynthesizerResult,
    board: Blackboard,
) -> CandidateRootCause:
    for result in (holistic, mechanical):
        for candidate in result.candidates:
            if candidate.component.strip().lower() == key:
                return candidate
    display = next((name for name, _ in board.top_components(30) if name.strip().lower() == key), key)
    evidence = board.evidence_for(display, limit=6)
    return CandidateRootCause(
        component=display,
        reason="unknown",
        confidence=0.0,
        score=board.component_score(display),
        evidence=[item.evidence_ptr for item in evidence if item.evidence_ptr],
        rationale="Proposed by the label-safe metric-shift specialist.",
    )


def _compact_candidates(result: SynthesizerResult, limit: int) -> list[dict[str, Any]]:
    return [candidate.compact() for candidate in result.candidates[:limit]]


def _compact_peer_reviews(transcript: list[dict[str, Any]], eligible: list[str]) -> list[dict[str, Any]]:
    allowed = {component.strip().lower() for component in eligible}
    out = []
    for message in transcript:
        if message.get("type") != "peer_review":
            continue
        component = str(message.get("component") or "").strip()
        if component.lower() not in allowed:
            continue
        out.append({
            "from": message.get("from"),
            "to": message.get("to"),
            "component": component,
            "verdict": message.get("verdict"),
            "support": message.get("support", 0.0),
            "refute": message.get("refute", 0.0),
            "evidence_ptrs": list(message.get("evidence_ptrs") or [])[:4],
            "rationale": str(message.get("rationale") or "")[:240],
        })
        if len(out) >= 30:
            break
    return out


def _graph_context(graph: DependencyGraph | None, eligible: list[str]) -> dict[str, Any]:
    if graph is None:
        return {"available": False, "candidates": {}}
    return {
        "available": True,
        "candidates": {
            component: {
                "calls": sorted(graph.calls.get(component.strip().lower(), set()))[:8],
                "called_by": sorted(graph.called_by.get(component.strip().lower(), set()))[:8],
            }
            for component in eligible
        },
    }


def _dedupe_metric_components(shifts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for item in shifts:
        key = str(item["component"]).strip().lower()
        if key and key not in seen:
            out.append(item)
            seen.add(key)
    return out

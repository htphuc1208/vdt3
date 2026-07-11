"""Autonomous peer interaction layer for ShardRCA.

This module is the difference between a sharded pipeline and a stronger MAS:
workers first publish local hypotheses, then inspect peer hypotheses, challenge
or support them from their own evidence, and finally revise local posteriors.
The implementation is deliberately label-safe: agents exchange candidates,
evidence pointers, missing-evidence notes, and short rationales only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .board import Blackboard, CandidateEvidence, WorkerDistribution


INTERACTION_SYSTEM = """You are one autonomous ShardRCA investigator in a peer-review round.
You may use only your own local candidates/findings plus peer messages.
Do not request labels or hidden answers. Do not invent evidence pointers.
Return ONLY JSON:
{"messages": [
  {"to": "agent_id", "verdict": "support|challenge|abstain",
   "component": "candidate component", "reason": "reason family",
   "support": 0.0, "refute": 0.0, "evidence_ptrs": ["..."],
   "rationale": "short local critique"}
 ],
 "revised_candidates": [
  {"component": "component", "reason": "reason family",
   "probability": 0.0, "support": 0.0, "refute": 0.0,
   "evidence_ptrs": ["..."], "rationale": "short"}
 ],
 "other_mass": 0.0}"""

INTERACTION_TEMPERATURE = 0.0


@dataclass
class InteractionResult:
    evidence: list[CandidateEvidence] = field(default_factory=list)
    distributions: list[WorkerDistribution] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def interact_candidate_evidence(
    evidence: list[CandidateEvidence],
    board: Blackboard,
    *,
    llm: LLMClient | None = None,
    max_rounds: int = 1,
) -> InteractionResult:
    """Run peer interaction over local candidate evidence records.

    The returned evidence includes the original local hypotheses plus peer-review
    support/challenge records. With ``llm=None`` this is a deterministic
    auditable interaction; with an LLM, each agent may revise its local posterior.
    """

    grouped = _group_evidence(evidence)
    if len(grouped) < 2 or max_rounds <= 0:
        return InteractionResult(
            evidence=list(evidence),
            transcript=_proposal_messages(grouped),
            diagnostics={"enabled": False, "reason": "fewer than two autonomous agents"},
        )

    transcript = _proposal_messages(grouped)
    review_evidence: list[CandidateEvidence] = []
    usage = UsageStats()
    for agent_id in sorted(grouped):
        peers = _peer_proposals(grouped, exclude=agent_id)
        if llm is None:
            messages, local_reviews = _deterministic_candidate_reviews(agent_id, grouped[agent_id], peers)
        else:
            messages, local_reviews, llm_usage = _llm_candidate_reviews(
                agent_id, grouped[agent_id], peers, board, llm
            )
            usage = usage.add(llm_usage)
            if not messages and not local_reviews:
                messages, local_reviews = _deterministic_candidate_reviews(agent_id, grouped[agent_id], peers)
        transcript.extend(messages)
        review_evidence.extend(local_reviews)

    out = list(evidence) + review_evidence
    return InteractionResult(
        evidence=out,
        transcript=transcript,
        usage=usage,
        diagnostics={
            "enabled": True,
            "rounds": max_rounds,
            "agent_count": len(grouped),
            "message_count": len(transcript),
            "review_evidence_count": len(review_evidence),
            "review_policy": "symmetric_evidence_budget",
            "llm_safety_envelope": llm is not None,
            "llm_temperature": INTERACTION_TEMPERATURE if llm is not None else None,
            "llm_seed": llm.settings.seed if llm is not None else None,
        },
    )


def interact_worker_distributions(
    distributions: list[WorkerDistribution],
    board: Blackboard,
    *,
    components: list[str],
    reasons: list[str],
    llm: LLMClient | None = None,
    max_rounds: int = 1,
) -> InteractionResult:
    """Run peer interaction over worker posterior distributions.

    Each worker receives peer top hypotheses and emits a revised distribution.
    The original distributions are preserved in artifacts by the caller; the
    returned distributions are what the fusion layer should consume.
    """

    active = [dist for dist in distributions if dist.candidate_scope]
    if len(active) < 2 or max_rounds <= 0:
        return InteractionResult(
            distributions=list(distributions),
            transcript=_distribution_proposals(active),
            diagnostics={"enabled": False, "reason": "fewer than two autonomous agents"},
        )

    transcript = _distribution_proposals(active)
    revised: list[WorkerDistribution] = []
    usage = UsageStats()
    for dist in active:
        peers = _peer_distribution_proposals(active, exclude=dist.worker_id)
        if llm is None:
            messages, new_dist = _deterministic_distribution_revision(dist, peers, components, reasons)
        else:
            messages, new_dist, llm_usage = _llm_distribution_revision(
                dist, peers, board, components, reasons, llm
            )
            usage = usage.add(llm_usage)
            if new_dist is None:
                messages, new_dist = _deterministic_distribution_revision(dist, peers, components, reasons)
        transcript.extend(messages)
        revised.append(new_dist)

    missing = [dist for dist in distributions if dist.worker_id not in {item.worker_id for item in active}]
    out = revised + missing
    return InteractionResult(
        distributions=out,
        transcript=transcript,
        usage=usage,
        diagnostics={
            "enabled": True,
            "rounds": max_rounds,
            "agent_count": len(active),
            "message_count": len(transcript),
            "revised_distribution_count": len(revised),
            "llm_temperature": INTERACTION_TEMPERATURE if llm is not None else None,
            "llm_seed": llm.settings.seed if llm is not None else None,
        },
    )


def _group_evidence(evidence: list[CandidateEvidence]) -> dict[str, list[CandidateEvidence]]:
    grouped: dict[str, list[CandidateEvidence]] = {}
    for index, item in enumerate(evidence):
        agent_id = item.worker_id or item.shard_id or f"agent_{index}"
        grouped.setdefault(agent_id, []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: (float(item.support_score) - float(item.refute_score), -item.local_rank), reverse=True)
    return grouped


def _proposal_messages(grouped: dict[str, list[CandidateEvidence]]) -> list[dict[str, Any]]:
    out = []
    for agent_id, items in sorted(grouped.items()):
        top = items[0] if items else None
        out.append({
            "round": 0,
            "from": agent_id,
            "to": "all",
            "type": "proposal",
            "candidate": top.compact() if top else None,
        })
    return out


def _peer_proposals(
    grouped: dict[str, list[CandidateEvidence]],
    *,
    exclude: str,
) -> list[dict[str, Any]]:
    peers = []
    for agent_id, items in sorted(grouped.items()):
        if agent_id == exclude or not items:
            continue
        peers.append({"agent_id": agent_id, "candidate": items[0].compact()})
    return peers


def _deterministic_candidate_reviews(
    agent_id: str,
    local: list[CandidateEvidence],
    peers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[CandidateEvidence]]:
    local_by_component = {item.component.strip().lower(): item for item in local}
    local_top = _strongest_positive(local)
    per_review_budget = _per_review_budget(local, len(peers))
    messages: list[dict[str, Any]] = []
    reviews: list[CandidateEvidence] = []
    for peer in peers:
        cand = peer["candidate"]
        component = str(cand.get("component") or "")
        reason = str(cand.get("reason_family") or cand.get("reason") or "unknown")
        local_match = local_by_component.get(component.strip().lower())
        support = 0.0
        refute = 0.0
        if local_match is not None and _signed_evidence(local_match) > 0.0:
            support = min(_positive_evidence(local_match), per_review_budget)
            verdict = "support"
            evidence_ptrs = list(local_match.evidence_ptrs)[:6]
            rationale = f"{agent_id} independently observes evidence for peer component {component}."
            reviews.append(local_match.model_copy(update={
                "support_score": support,
                "refute_score": 0.0,
                "worker_id": f"{agent_id}_peer_review",
                "shard_id": agent_id,
                "local_rank": 1,
                "rationale": rationale,
            }))
        elif local_match is not None and _signed_evidence(local_match) < 0.0:
            refute = min(abs(_signed_evidence(local_match)), per_review_budget)
            verdict = "challenge"
            evidence_ptrs = list(local_match.evidence_ptrs)[:6]
            rationale = f"{agent_id} has direct local refutation for peer component {component}."
            reviews.append(local_match.model_copy(update={
                "support_score": 0.0,
                "refute_score": refute,
                "worker_id": f"{agent_id}_peer_review",
                "shard_id": agent_id,
                "local_rank": 1,
                "rationale": rationale,
            }))
        else:
            verdict = "abstain"
            evidence_ptrs = []
            rationale = (
                f"{agent_id} has no direct same-component evidence; disjoint shard scope "
                "is not counter-evidence."
            )
        messages.append({
            "round": 1,
            "from": agent_id,
            "to": peer["agent_id"],
            "type": "peer_review",
            "verdict": verdict,
            "component": component,
            "reason": reason,
            "support": round(support, 6),
            "refute": round(refute, 6),
            "evidence_ptrs": evidence_ptrs,
            "rationale": rationale,
        })
    return messages, reviews


def _signed_evidence(item: CandidateEvidence) -> float:
    return float(item.support_score) - float(item.refute_score)


def _positive_evidence(item: CandidateEvidence) -> float:
    return max(0.0, _signed_evidence(item))


def _strongest_positive(items: list[CandidateEvidence]) -> CandidateEvidence | None:
    positives = [item for item in items if _positive_evidence(item) > 0.0]
    if not positives:
        return None
    return max(positives, key=lambda item: (_positive_evidence(item), -int(item.local_rank)))


def _per_review_budget(local: list[CandidateEvidence], peer_count: int) -> float:
    """Bound one agent's total deterministic critique by its own evidence mass."""

    if peer_count <= 0:
        return 0.0
    total_positive = sum(_positive_evidence(item) for item in local)
    return total_positive / peer_count


def _llm_candidate_reviews(
    agent_id: str,
    local: list[CandidateEvidence],
    peers: list[dict[str, Any]],
    board: Blackboard,
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], list[CandidateEvidence], UsageStats]:
    allowed_ptrs = {ptr for item in local for ptr in item.evidence_ptrs}
    payload = {
        "agent_id": agent_id,
        "local_candidates": [item.compact() for item in local[:8]],
        "peer_proposals": peers,
        "local_findings": _findings_for_allowed_ptrs(board, allowed_ptrs),
        "rule": "Use only local candidates/findings and copied evidence pointers.",
    }
    response = llm.chat(
        [
            {"role": "system", "content": INTERACTION_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        force_json=True,
        temperature=INTERACTION_TEMPERATURE,
    )
    data = extract_json(response.content)
    evidence_budget = sum(_positive_evidence(item) for item in local)
    messages = _validated_messages(agent_id, data.get("messages", []), allowed_ptrs, evidence_budget=evidence_budget)
    reviews = _reviews_from_revised_candidates(
        agent_id,
        data.get("revised_candidates", []),
        local,
        allowed_ptrs,
        evidence_budget=evidence_budget,
    )
    return messages, reviews, response.usage


def _distribution_proposals(distributions: list[WorkerDistribution]) -> list[dict[str, Any]]:
    out = []
    for dist in distributions:
        top = dist.candidates[0] if dist.candidates else None
        out.append({
            "round": 0,
            "from": dist.worker_id,
            "to": "all",
            "type": "proposal",
            "modality": dist.modality,
            "candidate": top.compact() if top else None,
            "other_mass": round(float(dist.other_mass), 6),
        })
    return out


def _peer_distribution_proposals(
    distributions: list[WorkerDistribution],
    *,
    exclude: str,
) -> list[dict[str, Any]]:
    peers = []
    for dist in distributions:
        if dist.worker_id == exclude or not dist.candidates:
            continue
        peers.append({
            "agent_id": dist.worker_id,
            "modality": dist.modality,
            "candidate": dist.candidates[0].compact(),
        })
    return peers


def _deterministic_distribution_revision(
    dist: WorkerDistribution,
    peers: list[dict[str, Any]],
    components: list[str],
    reasons: list[str],
) -> tuple[list[dict[str, Any]], WorkerDistribution]:
    allowed_components = set(components)
    allowed_reasons = set(reasons)
    by_pair: dict[tuple[str, str], CandidateEvidence] = {}
    messages = []
    per_review_budget = _per_review_budget(dist.candidates, len(peers))
    for candidate in dist.candidates:
        pair = (candidate.component, candidate.reason_family)
        if candidate.component in allowed_components and candidate.reason_family in allowed_reasons:
            by_pair[pair] = candidate.model_copy()

    for peer in peers:
        cand = peer["candidate"]
        pair = (str(cand.get("component") or ""), str(cand.get("reason_family") or cand.get("reason") or ""))
        if pair[0] not in allowed_components or pair[1] not in allowed_reasons:
            continue
        local = by_pair.get(pair)
        verdict = "abstain"
        support = 0.0
        refute = 0.0
        if local is not None and _signed_evidence(local) > 0.0:
            support = min(_positive_evidence(local), per_review_budget)
            verdict = "support"
        elif local is not None and _signed_evidence(local) < 0.0:
            refute = min(abs(_signed_evidence(local)), per_review_budget)
            by_pair[pair] = local.model_copy(update={
                "support_score": 0.0,
                "refute_score": max(float(local.refute_score), refute),
                "rationale": (local.rationale + " Local evidence refutes this peer hypothesis.").strip(),
            })
            verdict = "challenge"
        messages.append({
            "round": 1,
            "from": dist.worker_id,
            "to": peer["agent_id"],
            "type": "peer_review",
            "verdict": verdict,
            "component": pair[0],
            "reason": pair[1],
            "support": round(support, 6),
            "refute": round(refute, 6),
        })

    revised = _renormalize_distribution(dist, list(by_pair.values()))
    return messages, revised


def _llm_distribution_revision(
    dist: WorkerDistribution,
    peers: list[dict[str, Any]],
    board: Blackboard,
    components: list[str],
    reasons: list[str],
    llm: LLMClient,
) -> tuple[list[dict[str, Any]], WorkerDistribution | None, UsageStats]:
    allowed_ptrs = {ptr for candidate in dist.candidates for ptr in candidate.evidence_ptrs}
    payload = {
        "agent_id": dist.worker_id,
        "modality": dist.modality,
        "candidate_scope": dist.candidate_scope,
        "allowed_reasons": reasons,
        "local_distribution": dist.compact(),
        "peer_proposals": peers,
        "local_findings": _findings_for_allowed_ptrs(board, allowed_ptrs),
    }
    response = llm.chat(
        [
            {"role": "system", "content": INTERACTION_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        force_json=True,
        temperature=INTERACTION_TEMPERATURE,
    )
    data = extract_json(response.content)
    evidence_budget = sum(_positive_evidence(item) for item in dist.candidates)
    messages = _validated_messages(
        dist.worker_id,
        data.get("messages", []),
        allowed_ptrs,
        evidence_budget=evidence_budget,
    )
    candidates = _reviews_from_revised_candidates(
        dist.worker_id,
        data.get("revised_candidates", []),
        dist.candidates,
        allowed_ptrs,
        components=set(components) & set(dist.candidate_scope),
        reasons=set(reasons),
        evidence_budget=evidence_budget,
    )
    if not candidates:
        return messages, None, response.usage
    try:
        other_mass = max(0.0, min(1.0, float(data.get("other_mass") or 0.0)))
    except Exception:
        other_mass = 0.0
    revised = _renormalize_distribution(dist, candidates, other_mass=other_mass)
    return messages, revised, response.usage


def _validated_messages(
    agent_id: str,
    raw_messages: Any,
    allowed_ptrs: set[str],
    *,
    evidence_budget: float | None = None,
) -> list[dict[str, Any]]:
    messages = []
    budget = max(0.0, float(evidence_budget)) if evidence_budget is not None else None
    for raw in raw_messages if isinstance(raw_messages, list) else []:
        if not isinstance(raw, dict):
            continue
        verdict = str(raw.get("verdict") or "abstain")
        if verdict not in {"support", "challenge", "abstain"}:
            verdict = "abstain"
        ptrs = [str(ptr) for ptr in raw.get("evidence_ptrs", []) if str(ptr) in allowed_ptrs]
        support = max(0.0, _float(raw.get("support")))
        refute = max(0.0, _float(raw.get("refute")))
        if budget is not None:
            support = min(support, budget)
            refute = min(refute, budget)
        if not ptrs:
            support = 0.0
            refute = 0.0
        messages.append({
            "round": 1,
            "from": agent_id,
            "to": str(raw.get("to") or "all"),
            "type": "peer_review",
            "verdict": verdict,
            "component": str(raw.get("component") or ""),
            "reason": str(raw.get("reason") or raw.get("reason_family") or "unknown"),
            "support": support,
            "refute": refute,
            "evidence_ptrs": ptrs,
            "rationale": str(raw.get("rationale") or ""),
        })
    return messages


def _reviews_from_revised_candidates(
    agent_id: str,
    raw_candidates: Any,
    fallback: list[CandidateEvidence],
    allowed_ptrs: set[str],
    *,
    components: set[str] | None = None,
    reasons: set[str] | None = None,
    evidence_budget: float | None = None,
) -> list[CandidateEvidence]:
    components = components or {item.component for item in fallback}
    reasons = reasons or {item.reason_family for item in fallback}
    out = []
    budget = max(0.0, float(evidence_budget)) if evidence_budget is not None else None
    fallback_by_pair = {(item.component, item.reason_family): item for item in fallback}
    for raw in raw_candidates if isinstance(raw_candidates, list) else []:
        if not isinstance(raw, dict):
            continue
        component = str(raw.get("component") or "")
        reason = str(raw.get("reason") or raw.get("reason_family") or "unknown")
        if component not in components or reason not in reasons:
            continue
        base = fallback_by_pair.get((component, reason))
        ptrs = [str(ptr) for ptr in raw.get("evidence_ptrs", []) if str(ptr) in allowed_ptrs]
        if base is None and not ptrs:
            continue
        support = max(0.0, _float(raw.get("support")))
        refute = max(0.0, _float(raw.get("refute")))
        if budget is not None:
            support = min(support, budget)
            refute = min(refute, budget)
        if not ptrs and base is None:
            support = 0.0
            refute = 0.0
        out.append(CandidateEvidence(
            component=component,
            reason_family=reason,
            probability=max(0.0, min(1.0, _float(raw.get("probability")))),
            support_score=support,
            refute_score=refute,
            modality=base.modality if base is not None else "auxiliary",
            worker_id=agent_id,
            shard_id=agent_id,
            evidence_ptrs=ptrs or (list(base.evidence_ptrs)[:6] if base is not None else []),
            local_rank=len(out) + 1,
            rationale=str(raw.get("rationale") or "Revised after autonomous peer interaction."),
        ))
    return out


def _renormalize_distribution(
    original: WorkerDistribution,
    candidates: list[CandidateEvidence],
    *,
    other_mass: float | None = None,
) -> WorkerDistribution:
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.probability),
            float(item.support_score) - float(item.refute_score),
        ),
        reverse=True,
    )[:8]
    raw_scores = [
        max(0.0, float(item.probability)) + max(0.0, float(item.support_score) - float(item.refute_score)) * 0.01
        for item in ranked
    ]
    total = sum(raw_scores)
    explicit_mass = 0.9 if total > 0 else 0.0
    if other_mass is not None:
        explicit_mass = max(0.0, min(1.0, 1.0 - other_mass))
    revised = []
    for rank, (candidate, score) in enumerate(zip(ranked, raw_scores), start=1):
        probability = explicit_mass * score / total if total > 0 else 0.0
        revised.append(candidate.model_copy(update={
            "probability": probability,
            "worker_id": original.worker_id,
            "shard_id": original.worker_id,
            "local_rank": rank,
        }))
    return WorkerDistribution(
        worker_id=original.worker_id,
        modality=original.modality,
        candidate_scope=list(original.candidate_scope),
        candidates=revised,
        other_mass=max(0.0, 1.0 - sum(item.probability for item in revised)),
        notes=(original.notes + " autonomous_peer_interaction").strip(),
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _findings_for_allowed_ptrs(board: Blackboard, allowed_ptrs: set[str], *, limit: int = 16) -> list[dict[str, Any]]:
    """Render only findings owned by the reviewing worker's evidence shard."""

    return [
        finding.compact()
        for finding in board.ranked_findings()
        if finding.evidence_ptr and finding.evidence_ptr in allowed_ptrs
    ][:limit]

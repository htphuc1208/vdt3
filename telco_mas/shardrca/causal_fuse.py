"""LLM causal fusion head for ShardRCA (Hướng-2 fix).

Group-A diagnosis: the sharded path surfaces the true root in its top-k fused
candidates on 22/23 of its errors, but the *mechanical* fusion argmax ranks a
topology-adjacent symptom carrier (caller / hub) above it. The winning single
reader (`no_shard`) differs only in that an LLM reasons over the merged evidence.

This module adds that missing decision stage WITHOUT re-reading raw telemetry: it
reasons only over the compact fused-candidate summaries plus the dependency-graph
relationships (who calls whom, which nodes are symptomatic), so it keeps the
bounded-context benefit of sharding while applying the causal "blame the callee"
principle that mechanical fusion lacks. It can only pick among candidates the
deterministic layer already surfaced, so it stays label-safe and grounded.
"""
from __future__ import annotations

import json

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .board import Blackboard, CandidateRootCause
from .synthesizer import SynthesizerResult
from .topology import DependencyGraph


CAUSAL_FUSE_SYSTEM = """You are the ShardRCA causal fusion head.
You are given a SHORTLIST of candidate root-cause components (already extracted
from label-safe telemetry) and a service dependency graph. A fault propagates
from the faulty component UP to the services that call it, so callers and hubs
show *propagated* symptoms while the true root is usually the deepest still-anomalous
callee that explains the others.

Rules:
- Choose the root ONLY from the provided candidate components. Do not invent names.
- Apply "blame the callee": if A calls B and both look anomalous, prefer B (the callee).
- Prefer the candidate whose failure explains the most other symptomatic candidates.
- Do not prefer a highly-connected hub (e.g. frontend) just because it shows many symptoms.
Return ONLY JSON:
{"root": "candidate component", "reason": "cpu|mem|disk|delay|loss|socket|network|db|unknown",
 "ranked_roots": ["c1","c2","c3"], "confidence": 0.0, "rationale": "short causal justification"}"""


def _candidate_payload(
    synthesis: SynthesizerResult,
    graph: DependencyGraph | None,
    symptomatic: set[str],
    max_candidates: int,
) -> list[dict]:
    sym = {s.strip().lower() for s in symptomatic}
    payload = []
    for cand in synthesis.candidates[:max_candidates]:
        key = cand.component.strip().lower()
        callers = sorted(graph.called_by.get(key, set())) if graph else []
        callees = sorted(graph.calls.get(key, set())) if graph else []
        payload.append({
            "component": cand.component,
            "reason": cand.reason,
            "fusion_score": round(float(cand.score), 6),
            "is_symptomatic": key in sym,
            "calls": callees[:8],           # this candidate depends on these (its callees)
            "called_by": callers[:8],       # these depend on this candidate (its callers)
            "in_graph": bool(graph and key in graph.nodes),
            "evidence": list(cand.evidence)[:5],
        })
    return payload


def llm_causal_fuse(
    synthesis: SynthesizerResult,
    board: Blackboard,
    graph: DependencyGraph | None,
    symptomatic: set[str],
    *,
    llm: LLMClient | None,
    k: int = 3,
    max_candidates: int = 8,
) -> SynthesizerResult:
    """Re-decide the winner among fused candidates using an LLM with causal hints.

    Falls back to the input synthesis when there is no LLM, fewer than two
    candidates, or the LLM never returns a valid in-shortlist component.
    """
    if llm is None or len(synthesis.candidates) < 2:
        return synthesis

    allowed = {c.component.strip().lower(): c for c in synthesis.candidates}
    payload = {
        "candidates": _candidate_payload(synthesis, graph, symptomatic, max_candidates),
        "instruction": "Pick the single most likely ROOT cause from candidates using the graph.",
    }
    usage = UsageStats()
    votes: dict[str, float] = {}
    best_by_component: dict[str, CandidateRootCause] = {}
    ranked_accumulator: dict[str, float] = {}
    for _ in range(max(1, k)):
        response = llm.chat(
            [
                {"role": "system", "content": CAUSAL_FUSE_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
            ],
            force_json=True,
        )
        usage = usage.add(response.usage)
        data = extract_json(response.content)
        root = str(data.get("root") or "").strip().lower()
        if root not in allowed:
            continue
        try:
            conf = float(data.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        votes[root] = votes.get(root, 0.0) + 1.0 + max(0.0, min(1.0, conf))
        base = allowed[root]
        best_by_component[root] = base.model_copy(update={
            "reason": str(data.get("reason") or base.reason),
            "rationale": str(data.get("rationale") or base.rationale),
        })
        # accumulate ranked order for tie-breaking / ranked_roots
        ranked = data.get("ranked_roots") if isinstance(data.get("ranked_roots"), list) else []
        for rank, name in enumerate(ranked):
            nk = str(name).strip().lower()
            if nk in allowed:
                ranked_accumulator[nk] = ranked_accumulator.get(nk, 0.0) + (len(ranked) - rank)

    if not votes:
        return synthesis

    winner_key = max(votes, key=lambda k_: (votes[k_], ranked_accumulator.get(k_, 0.0)))
    # Re-rank candidates: LLM winner first, then by accumulated ranked votes, then fusion score.
    def _rank_key(cand: CandidateRootCause):
        key = cand.component.strip().lower()
        return (
            key == winner_key,
            ranked_accumulator.get(key, 0.0),
            float(cand.score),
        )
    reordered = sorted(synthesis.candidates, key=_rank_key, reverse=True)
    winner = best_by_component.get(winner_key, reordered[0])
    # keep winner object at head
    reordered = [winner] + [c for c in reordered if c.component.strip().lower() != winner_key]
    return SynthesizerResult(
        winner=winner,
        candidates=reordered,
        vote_breakdown={comp: round(v, 4) for comp, v in votes.items()},
        usage=synthesis.usage.add(usage),
    )

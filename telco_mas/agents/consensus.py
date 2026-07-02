"""Consensus module — fuse the experts' hypotheses into one decision.

Inspired by mABC (multi-Agent Blockchain-inspired Collaboration) and multi-agent
consensus theory: each expert is a "node" that casts a confidence-weighted vote for a
root-cause element; agreeing nodes reinforce each other. A lightweight LLM *arbiter*
then reviews the tally and the experts' evidence and issues the final, explained verdict.
"""
from __future__ import annotations

import json

from ..schemas import ConsensusResult, Hypothesis, Incident, UsageStats
from .base import BaseAgent, as_float, as_str, incident_brief

AGREEMENT_BONUS = 0.15


def tally_votes(hypotheses: list[Hypothesis]) -> tuple[dict[str, float], dict[str, list[Hypothesis]]]:
    """Weighted vote per candidate element + the hypotheses backing each."""
    scores: dict[str, float] = {}
    backers: dict[str, list[Hypothesis]] = {}
    for h in hypotheses:
        el = h.faulty_element_id or "UNKNOWN"
        scores[el] = scores.get(el, 0.0) + h.confidence
        backers.setdefault(el, []).append(h)
    for el, hs in backers.items():
        distinct_experts = len({h.proposed_by for h in hs})
        scores[el] += AGREEMENT_BONUS * (distinct_experts - 1)
    return scores, backers


ARBITER_SYSTEM = """You are the incident-bridge arbiter. Several domain experts each proposed a
root-cause hypothesis, and a confidence-weighted vote has been tallied. Decide the single most
likely root cause.

Prefer the candidate with the strongest combined evidence and votes, but you MAY override the tally
if one expert's concrete evidence clearly outweighs the others (explain why). Resolve conflicts:
when experts disagree, the correct answer is usually the *upstream* element whose failure explains
the most downstream symptoms.

Respond with ONLY a JSON object:
{"faulty_element_id": "...", "fault_type": "...", "root_cause": "one sentence",
 "confidence": 0.0-1.0, "explanation": "how you weighed the experts and resolved any conflict"}"""


class ConsensusModule(BaseAgent):
    name = "consensus"
    tool_names: list[str] = []  # pure reasoning over the experts' outputs

    def run(self, incident: Incident, hypotheses: list[Hypothesis]):
        scores, backers = tally_votes(hypotheses)
        ranked = sorted(hypotheses, key=lambda h: scores.get(h.faulty_element_id or "UNKNOWN", 0.0), reverse=True)

        expert_block = "\n".join(
            f"- {h.proposed_by}: element={h.faulty_element_id}, type={h.fault_type}, "
            f"confidence={h.confidence:.2f}\n    root_cause: {h.root_cause}\n    evidence: {', '.join(h.evidence) or 'n/a'}"
            for h in hypotheses
        )
        tally_block = json.dumps({k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])})
        user = (
            incident_brief(incident)
            + "\n\nExpert hypotheses:\n" + expert_block
            + "\n\nWeighted vote tally (element -> score):\n" + tally_block
        )
        run = self.invoke(ARBITER_SYSTEM, user)
        d = run.data

        # fall back to the top of the tally if the arbiter is unclear
        top_el = max(scores, key=scores.get) if scores else None
        top_hyp = next((h for h in ranked if h.faulty_element_id == top_el), ranked[0] if ranked else None)
        faulty = as_str(d.get("faulty_element_id")) or top_el
        fault_type = as_str(d.get("fault_type")) or (top_hyp.fault_type if top_hyp else None)
        root_cause = as_str(d.get("root_cause")) or (top_hyp.root_cause if top_hyp else "Undetermined")
        total = sum(scores.values()) or 1.0
        confidence = as_float(d.get("confidence"), round(scores.get(faulty, 0.0) / total, 2)) if faulty else 0.0

        result = ConsensusResult(
            root_cause=root_cause,
            faulty_element_id=faulty,
            fault_type=fault_type,
            confidence=confidence,
            ranked=ranked,
            vote_breakdown={k: round(v, 3) for k, v in scores.items()},
            explanation=as_str(d.get("explanation")),
        )
        return result, run.trace, run.usage or UsageStats()

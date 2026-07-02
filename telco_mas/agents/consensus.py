"""Consensus module — fuse expert hypotheses into one decision.

The mechanism is an mABC-inspired engineering heuristic: each expert casts a
calibrated, evidence-weighted vote. An LLM arbiter is used only when the margin
is low or concrete evidence conflicts, keeping the common case auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..schemas import ConsensusResult, Hypothesis, Incident, UsageStats
from .base import BaseAgent, as_float, as_str, incident_brief

AGREEMENT_BONUS = 0.15
LOW_MARGIN = 0.20
ABSTAIN_THRESHOLD = 0.35


@dataclass(frozen=True)
class Calibration:
    """Simple confidence calibration fallback.

    In paper runs this can be replaced by held-out calibration artifacts. The
    default shrinks overconfident LLM self-scores toward 0.5, following the
    calibration concern in Guo et al. (2017).
    """

    slope: float = 0.85
    bias: float = 0.0

    def apply(self, confidence: float) -> float:
        return min(1.0, max(0.0, 0.5 + (confidence - 0.5) * self.slope + self.bias))


DEFAULT_CALIBRATION: dict[str, Calibration] = {
    "ran_expert": Calibration(0.82, 0.0),
    "transport_expert": Calibration(0.86, 0.0),
    "core_expert": Calibration(0.84, 0.0),
}


def tally_votes(
    hypotheses: list[Hypothesis],
    calibration: dict[str, Calibration] | None = None,
) -> tuple[dict[str, float], dict[str, list[Hypothesis]]]:
    """Calibrated evidence-weighted vote per candidate element."""
    calibration = calibration or DEFAULT_CALIBRATION
    scores: dict[str, float] = {}
    backers: dict[str, list[Hypothesis]] = {}
    for h in hypotheses:
        el = h.faulty_element_id or "UNKNOWN"
        cal = calibration.get(h.proposed_by, Calibration())
        vote = cal.apply(h.confidence) + _evidence_bonus(h) + _topology_bonus(h) + _rag_bonus(h)
        scores[el] = scores.get(el, 0.0) + round(vote, 4)
        backers.setdefault(el, []).append(h)
    for el, hs in backers.items():
        distinct_experts = len({h.proposed_by for h in hs})
        scores[el] += AGREEMENT_BONUS * (distinct_experts - 1)
    return {k: round(v, 4) for k, v in scores.items()}, backers


def _evidence_bonus(h: Hypothesis) -> float:
    concrete = 0
    joined = " ".join(h.evidence).lower()
    for token in ("diagnostic", "rx", "dbm", "cpu", "crc", "servfail", "battery", "bler", "license", "config"):
        if token in joined:
            concrete += 1
    concrete += min(3, len(h.evidence))
    return min(0.25, 0.04 * concrete)


def _topology_bonus(h: Hypothesis) -> float:
    text = f"{h.rationale} {h.root_cause}".lower()
    if any(token in text for token in ("upstream", "dependency", "downstream", "blast radius", "parent")):
        return 0.10
    return 0.0


def _rag_bonus(h: Hypothesis) -> float:
    text = " ".join([h.rationale, h.root_cause, *h.evidence]).lower()
    if "sop-" in text or "hist-" in text or "historical" in text or "playbook" in text:
        return 0.08
    return 0.0


def _needs_arbiter(scores: dict[str, float], hypotheses: list[Hypothesis]) -> bool:
    if len(scores) <= 1:
        return False
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    disagree = len({h.faulty_element_id for h in hypotheses if h.faulty_element_id}) > 1
    return disagree and margin < LOW_MARGIN


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

    def run(self, incident: Incident, hypotheses: list[Hypothesis], use_arbiter: bool = True):
        scores, backers = tally_votes(hypotheses)
        ranked = sorted(hypotheses, key=lambda h: scores.get(h.faulty_element_id or "UNKNOWN", 0.0), reverse=True)
        top_el = max(scores, key=scores.get) if scores else None
        top_hyp = next((h for h in ranked if h.faulty_element_id == top_el), ranked[0] if ranked else None)

        if not (use_arbiter and _needs_arbiter(scores, hypotheses)):
            total = sum(scores.values()) or 1.0
            confidence = round(scores.get(top_el or "UNKNOWN", 0.0) / total, 2) if top_el else 0.0
            if confidence < ABSTAIN_THRESHOLD:
                top_el = None
            result = ConsensusResult(
                root_cause=top_hyp.root_cause if top_hyp else "Undetermined",
                faulty_element_id=top_el,
                fault_type=top_hyp.fault_type if top_hyp else None,
                confidence=confidence,
                ranked=ranked,
                vote_breakdown={k: round(v, 3) for k, v in scores.items()},
                explanation=(
                    "Calibrated evidence-weighted consensus selected the top supported candidate; "
                    "arbiter skipped because the vote margin was sufficient."
                ),
            )
            return result, [], UsageStats()

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

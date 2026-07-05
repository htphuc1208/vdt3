"""Consensus module — fuse expert hypotheses into one decision.

mABC-inspired, with one deliberate difference from the earlier prototype: vote
weights come from **verifiable signals**, not from keyword-spotting the experts'
prose (which was gameable — an expert could earn weight by *mentioning* "dBm" or
"upstream" without having checked anything). The two verifiable signals are:

* ``verified_evidence`` — did this expert actually RUN a diagnostic on the element
  it blames, and did the environment return a non-nominal finding? (read from the
  tool-call trace, i.e. ground truth, not from the hypothesis text);
* ``topology_coverage`` — computed on the dependency graph: what fraction of the
  alarmed elements lies inside the blamed element's failure closure
  (itself + descendants + same site)?

This mirrors mABC's expertise/contribution indices: weight follows demonstrated,
checkable work. An LLM arbiter is consulted only when the vote margin is low.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from ..knowledge.fault_ontology import (
    FAULT_EXPLANATIONS,
    canonicalize_fault_type,
    direct_root_condition_families,
    domain_compatible,
    earliest_specific_root_candidates,
)
from ..schemas import ConsensusResult, Hypothesis, Incident, TraceStep, UsageStats
from .base import BaseAgent, as_float, as_str, incident_brief

AGREEMENT_BONUS = 0.15
VERIFIED_BONUS = 0.35
COVERAGE_WEIGHT = 0.30
CAUSAL_SUPPORT_WEIGHT = 0.65
LOW_MARGIN = 0.20
ABSTAIN_THRESHOLD = 0.35


@dataclass(frozen=True)
class Calibration:
    """Confidence shrinkage toward 0.5 (LLM self-scores are overconfident —
    Guo et al. 2017). Replaceable by held-out calibration artifacts."""

    slope: float = 0.85
    bias: float = 0.0

    def apply(self, confidence: float) -> float:
        return min(1.0, max(0.0, 0.5 + (confidence - 0.5) * self.slope + self.bias))


DEFAULT_CALIBRATION: dict[str, Calibration] = {
    "ran_expert": Calibration(0.82, 0.0),
    "transport_expert": Calibration(0.86, 0.0),
    "power_expert": Calibration(0.84, 0.0),
    "core_expert": Calibration(0.84, 0.0),
}


# --------------------------------------------------------------------------- #
# Verifiable signals
# --------------------------------------------------------------------------- #
def verified_diagnostic_findings(trace: list[TraceStep] | None, element_id: str | None) -> list[str]:
    """Diagnostic results this expert actually obtained on the element it blames.

    Environment-grounded: read from executed tool calls, not from claim text.
    A finding counts only if it is non-nominal and in-scope.
    """
    if not trace or not element_id:
        return []
    findings: list[str] = []
    for step in trace:
        for call in step.tool_calls:
            if call.name != "run_diagnostic":
                continue
            if str(call.arguments.get("element_id", "")) != element_id:
                continue
            preview = call.result_preview.lower()
            if "nominal" in preview or "outside your domain" in preview or "unknown element" in preview:
                continue
            findings.append(call.result_preview)
    return findings


def topology_coverage(topology, element_id: str | None, symptomatic: set[str]) -> float:
    """Fraction of alarmed elements explained by a fault on ``element_id``.

    Closure = the element itself, its dependency descendants, and co-site
    elements. Computed on the graph — not substring matching.
    """
    if not element_id or not symptomatic or element_id not in topology:
        return 0.0
    el = topology.get(element_id)
    closure = {element_id}
    closure |= {d.id for d in topology.descendants(element_id)}
    closure |= {e.id for e in topology.at_site(el.site)}
    return len(symptomatic & closure) / len(symptomatic)


def tally_votes(
    hypotheses: list[Hypothesis],
    calibration: dict[str, Calibration] | None = None,
    verified: dict[str, list[str]] | None = None,
    coverage: dict[str, float] | None = None,
    causal_support: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, list[Hypothesis]]]:
    """Verifiable-evidence-weighted vote per candidate element.

    ``verified``  maps expert name -> non-nominal diagnostic findings on its blamed element.
    ``coverage``  maps expert name -> topology coverage of its blamed element.
    Both default to empty (pure shrunk-confidence vote) for backward compatibility.
    """
    calibration = calibration or DEFAULT_CALIBRATION
    verified = verified or {}
    coverage = coverage or {}
    causal_support = causal_support or {}
    scores: dict[str, float] = {}
    backers: dict[str, list[Hypothesis]] = {}
    for h in hypotheses:
        el = h.faulty_element_id or "UNKNOWN"
        cal = calibration.get(h.proposed_by, Calibration())
        vote = cal.apply(h.confidence)
        support = causal_support.get(h.proposed_by, 0.0)
        if support <= -1.0:
            # A canonical fault family on an impossible equipment domain is
            # structurally invalid, regardless of model self-confidence.
            vote = 0.0
        else:
            if verified.get(h.proposed_by):
                vote += VERIFIED_BONUS
            vote += COVERAGE_WEIGHT * coverage.get(h.proposed_by, 0.0)
            vote += CAUSAL_SUPPORT_WEIGHT * support
            vote = max(0.0, vote)
        scores[el] = scores.get(el, 0.0) + round(vote, 4)
        backers.setdefault(el, []).append(h)
    for el, hs in backers.items():
        distinct_experts = len({h.proposed_by for h in hs})
        scores[el] += AGREEMENT_BONUS * (distinct_experts - 1)
    return {k: round(v, 4) for k, v in scores.items()}, backers


def _needs_arbiter(scores: dict[str, float], hypotheses: list[Hypothesis]) -> bool:
    if len(scores) <= 1:
        return False
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    disagree = len({h.faulty_element_id for h in hypotheses if h.faulty_element_id}) > 1
    return disagree and margin < LOW_MARGIN


ARBITER_SYSTEM = """You are the incident-bridge arbiter. Several domain experts each proposed a
root-cause hypothesis; a vote weighted by VERIFIED evidence (diagnostics they actually ran) and
topology coverage has been tallied, but the margin is low. Decide the single most likely root cause.

Weigh VERIFIED diagnostic findings above self-reported confidence. When experts disagree, prefer the
candidate whose failure explains the most downstream symptoms (upstream causes beat symptom nodes).

Respond with ONLY a JSON object:
{"faulty_element_id": "...", "fault_type": "...", "root_cause": "one sentence",
 "confidence": 0.0-1.0, "explanation": "how you weighed the experts and resolved the conflict"}"""


class ConsensusModule(BaseAgent):
    name = "consensus"
    tool_names: list[str] = []  # pure reasoning over the experts' outputs

    def run(
        self,
        incident: Incident,
        hypotheses: list[Hypothesis],
        expert_traces: Optional[dict[str, list[TraceStep]]] = None,
        use_arbiter: bool = True,
        use_evidence_verifier: bool = True,
    ):
        expert_traces = expert_traces or {}
        symptomatic = {a.element_id for a in incident.alarms}
        hypotheses = [
            _normalize_hypothesis(hypothesis, incident, self.ctx.sim.topology)
            for hypothesis in hypotheses
        ]
        if use_evidence_verifier:
            hypotheses += _missing_ontology_candidates(
                hypotheses,
                incident,
                self.ctx.sim.topology,
            )
        verified = {
            h.proposed_by: verified_diagnostic_findings(expert_traces.get(h.proposed_by), h.faulty_element_id)
            for h in hypotheses
        }
        coverage = {
            h.proposed_by: round(topology_coverage(self.ctx.sim.topology, h.faulty_element_id, symptomatic), 3)
            for h in hypotheses
        }
        causal_support = (
            {
                h.proposed_by: _causal_support(h, incident, self.ctx.sim.topology)
                for h in hypotheses
            }
            if use_evidence_verifier
            else {}
        )

        scores, backers = tally_votes(
            hypotheses,
            verified=verified,
            coverage=coverage,
            causal_support=causal_support,
        )
        ranked = sorted(hypotheses, key=lambda h: scores.get(h.faulty_element_id or "UNKNOWN", 0.0), reverse=True)
        top_el = max(scores, key=scores.get) if scores else None
        top_hyp = next((h for h in ranked if h.faulty_element_id == top_el), ranked[0] if ranked else None)

        signal_note = "; ".join(
            f"{h.proposed_by}: verified={'yes' if verified.get(h.proposed_by) else 'no'}, "
            f"coverage={coverage.get(h.proposed_by, 0.0):.2f}, "
            f"causal_support={causal_support.get(h.proposed_by, 0.0):.2f}"
            for h in hypotheses
        )

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
                    "Verifiable-evidence-weighted consensus selected the top candidate "
                    f"(arbiter skipped: margin sufficient). Signals — {signal_note}"
                ),
            )
            return result, [], UsageStats()

        expert_block = "\n".join(
            f"- {h.proposed_by}: element={h.faulty_element_id}, type={h.fault_type}, "
            f"confidence={h.confidence:.2f}, topology_coverage={coverage.get(h.proposed_by, 0.0):.2f}\n"
            f"    causal_support={causal_support.get(h.proposed_by, 0.0):.2f}\n"
            f"    root_cause: {h.root_cause}\n"
            f"    VERIFIED diagnostics on the blamed element: "
            f"{'; '.join(verified.get(h.proposed_by, [])) or 'NONE (never confirmed by a diagnostic)'}"
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
        fault_type = canonicalize_fault_type(
            fault_type,
            element=self.ctx.sim.topology.get(faulty) if faulty else None,
            alarms=incident.alarms,
            root_cause=root_cause,
        )
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


def _normalize_hypothesis(hypothesis: Hypothesis, incident: Incident, topology) -> Hypothesis:
    element = topology.get(hypothesis.faulty_element_id) if hypothesis.faulty_element_id else None
    canonical = canonicalize_fault_type(
        hypothesis.fault_type,
        element=element,
        alarms=incident.alarms,
        root_cause=hypothesis.root_cause,
    )
    if canonical == hypothesis.fault_type:
        return hypothesis
    return hypothesis.model_copy(update={"fault_type": canonical})


def _missing_ontology_candidates(
    hypotheses: list[Hypothesis],
    incident: Incident,
    topology,
) -> list[Hypothesis]:
    existing = {(item.faulty_element_id, item.fault_type) for item in hypotheses}
    candidates = []
    for element_id, family, alarm in earliest_specific_root_candidates(incident, topology):
        if (element_id, family) in existing:
            continue
        candidates.append(Hypothesis(
            proposed_by=f"evidence_verifier:{element_id}",
            faulty_element_id=element_id,
            fault_type=family,
            root_cause=FAULT_EXPLANATIONS[family],
            confidence=0.8,
            rationale=(
                "High-specificity root condition appeared at the first event timestamp; "
                "generic propagated symptom alarms are excluded from this rule."
            ),
            evidence=[
                f"{alarm.raised_at.isoformat() if alarm.raised_at else 'unknown-time'} "
                f"{alarm.element_id} {alarm.name}: {alarm.probable_cause}"
            ],
        ))
    return candidates


def _causal_support(hypothesis: Hypothesis, incident: Incident, topology) -> float:
    if not hypothesis.faulty_element_id:
        return -1.0
    element = topology.get(hypothesis.faulty_element_id)
    if element is None:
        return -1.0
    score = 0.0
    compatible = domain_compatible(hypothesis.fault_type, element)
    if compatible is False:
        score -= 1.0
    direct = direct_root_condition_families(element, incident.alarms)
    if hypothesis.fault_type in direct:
        score += 0.6
    earliest = {
        (element_id, family)
        for element_id, family, _alarm in earliest_specific_root_candidates(incident, topology)
    }
    if (hypothesis.faulty_element_id, hypothesis.fault_type) in earliest:
        score += 0.4
    return min(1.0, max(-1.0, score))

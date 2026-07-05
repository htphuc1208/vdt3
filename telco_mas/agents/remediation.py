"""Remediation agent: turn the confirmed root cause into a concrete SOP-based plan."""
from __future__ import annotations

from ..knowledge.fault_ontology import REMEDIATION_FAMILY_HINTS
from ..schemas import ConsensusResult, Incident, RemediationPlan, UsageStats
from .base import BaseAgent, as_list, as_str, incident_brief

SYSTEM = """You are the remediation agent. The root cause has been confirmed. Produce a concrete,
actionable remediation plan grounded in the operator's SOP playbooks.

Use `search_knowledge_base` to fetch the matching SOP, then adapt it to this incident. The `action`
field must be a single imperative sentence that names the fix and the target element, so it can be
executed (e.g. "Dispatch a field team to splice and repair FIBER-LINK-01", "Replace the faulty line
card on TRANSPORT-RTR-02", "Roll back the config change on CORE-AMF-01").
Do not remediate a downstream symptom family when the confirmed root family is more specific.
The fix-family hint is a constraint, not an exact hidden SOP; use the knowledge base for details.

Respond with ONLY a JSON object:
{"sop_id": "SOP-...",
 "summary": "one line",
 "steps": ["...", "..."],
 "expected_outcome": "what should recover",
 "action": "single imperative fix naming the target element",
 "target_element_id": "the element to fix"}"""


class RemediationAgent(BaseAgent):
    name = "remediation"
    tool_names = ["search_knowledge_base", "get_historical_incidents", "query_topology"]

    def run(
        self,
        incident: Incident,
        consensus: ConsensusResult,
        *,
        previous_failure: str | None = None,
    ):
        hint = REMEDIATION_FAMILY_HINTS.get(consensus.fault_type or "")
        user = (
            incident_brief(incident)
            + f"\n\nConfirmed root cause: {consensus.root_cause}"
            + f"\nFaulty element: {consensus.faulty_element_id} (type {consensus.fault_type})."
            + (f"\nRequired fix family: {hint}." if hint else "")
            + (
                "\n\nThe previous remediation had NO EFFECT. Do not repeat it. "
                f"Failure evidence: {previous_failure}"
                if previous_failure
                else ""
            )
        )
        run = self.invoke(SYSTEM, user)
        d = run.data
        plan = RemediationPlan(
            sop_id=as_str(d.get("sop_id")) or None,
            summary=as_str(d.get("summary")),
            steps=as_list(d.get("steps")),
            expected_outcome=as_str(d.get("expected_outcome")),
        )
        action = as_str(d.get("action")) or plan.summary or f"Apply {plan.sop_id} to {consensus.faulty_element_id}"
        target = as_str(d.get("target_element_id")) or consensus.faulty_element_id
        return plan, action, target, run.trace, run.usage or UsageStats()

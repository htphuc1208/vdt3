"""Validation agent: apply the fix (simulated) and verify the incident is resolved."""
from __future__ import annotations

from ..schemas import RemediationPlan, UsageStats, ValidationResult
from .base import BaseAgent, as_list, as_str

SYSTEM = """You are the validation agent. Execute the proposed remediation and verify it worked.

Steps:
1. Call `apply_remediation` with the given action and target element.
2. Re-check `query_alarms` and `query_kpis` to confirm the alarms cleared and KPIs returned to normal.
3. Decide whether the incident is truly resolved.

Respond with ONLY a JSON object:
{"resolved": true/false,
 "notes": "what you applied and what recovered (or why it did not)",
 "recovered_kpis": ["metric@element", ...]}"""


class ValidationAgent(BaseAgent):
    name = "validation"
    tool_names = ["apply_remediation", "query_alarms", "query_kpis"]

    def run(self, plan: RemediationPlan, action: str, target_element_id: str | None):
        user = (
            f"Proposed remediation action: {action}\n"
            f"Target element: {target_element_id}\n"
            f"SOP: {plan.sop_id}\nExpected outcome: {plan.expected_outcome}\n\n"
            "Apply it and verify."
        )
        run = self.invoke(SYSTEM, user)
        d = run.data
        resolved = bool(d.get("resolved", False))
        result = ValidationResult(
            resolved=resolved,
            notes=as_str(d.get("notes")),
            recovered_kpis=as_list(d.get("recovered_kpis")),
        )
        return result, run.trace, run.usage or UsageStats()

"""Detection / triage agent: classify severity and localise the suspected domain."""
from __future__ import annotations

from ..schemas import Domain, Incident, Severity, TriageResult, UsageStats
from .base import BaseAgent, as_list, as_str, incident_brief

SYSTEM = """You are a NOC (Network Operations Centre) triage agent for a 5G mobile network.
Given an incident (a burst of alarms), your job is to quickly assess it:
- overall severity,
- the single most likely affected DOMAIN (RAN, TRANSPORT, CORE, or POWER),
- the set of affected elements,
- a one-line summary.

Use the tools to inspect alarms, topology and KPIs. Remember that many alarms can share
ONE upstream root cause: follow the dependency chain (children depend on parents) to find the
smallest set of elements that could explain everything.

When done, respond with ONLY a JSON object:
{"severity": "CRITICAL|MAJOR|MINOR|WARNING",
 "suspected_domain": "RAN|TRANSPORT|CORE|POWER",
 "affected_elements": ["..."],
 "summary": "..."}"""


class DetectionAgent(BaseAgent):
    name = "detection"
    tool_names = ["query_alarms", "query_topology", "query_kpis"]

    def run(self, incident: Incident):
        run = self.invoke(SYSTEM, incident_brief(incident))
        d = run.data
        try:
            severity = Severity(as_str(d.get("severity"), "MAJOR").upper())
        except Exception:
            severity = Severity.MAJOR
        try:
            domain = Domain(as_str(d.get("suspected_domain"), "UNKNOWN").upper())
        except Exception:
            domain = Domain.UNKNOWN
        result = TriageResult(
            severity=severity,
            suspected_domain=domain,
            affected_elements=as_list(d.get("affected_elements")) or incident.affected_elements,
            summary=as_str(d.get("summary")),
        )
        return result, run.trace, run.usage or UsageStats()

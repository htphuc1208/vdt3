"""Correlation / knowledge agent: RAG over SOPs + historical incidents."""
from __future__ import annotations

from ..schemas import Incident, TriageResult, UsageStats
from .base import BaseAgent, as_list, as_str, incident_brief

SYSTEM = """You are a knowledge-correlation agent. You connect the current incident's symptoms to
the operator's knowledge base: Standard Operating Procedures (SOPs) and similar historical incidents.

Use `search_knowledge_base` and `get_historical_incidents` (and topology/alarms if useful) to find
the SOPs and past incidents whose symptoms best match this incident. Summarise what they suggest the
likely root cause and remediation direction are — this context will be handed to the diagnosis experts.

When done, respond with ONLY a JSON object:
{"notes": "concise correlation summary for the diagnosis team",
 "relevant_sops": ["SOP-..."],
 "similar_incidents": ["HIST-..."],
 "likely_domains": ["RAN|TRANSPORT|CORE|POWER", ...]}"""


class CorrelationAgent(BaseAgent):
    name = "correlation"
    tool_names = ["search_knowledge_base", "get_historical_incidents", "query_alarms", "query_topology"]

    def run(self, incident: Incident, triage: TriageResult):
        user = incident_brief(incident) + (
            f"\n\nTriage said: severity={triage.severity.value}, "
            f"suspected_domain={triage.suspected_domain.value}. {triage.summary}"
        )
        run = self.invoke(SYSTEM, user)
        d = run.data
        sops = as_list(d.get("relevant_sops"))
        hist = as_list(d.get("similar_incidents"))
        notes = as_str(d.get("notes"))
        summary = notes
        if sops:
            summary += f"\nRelevant SOPs: {', '.join(sops)}"
        if hist:
            summary += f"\nSimilar incidents: {', '.join(hist)}"
        return summary.strip(), run.trace, run.usage or UsageStats()

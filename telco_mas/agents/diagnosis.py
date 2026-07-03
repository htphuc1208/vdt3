"""Domain-expert diagnosis agents.

A *team* of specialists (RAN, Transport & Infrastructure, Core) each investigates the
incident from its own perspective and returns a ranked root-cause hypothesis with a
confidence and supporting evidence. Their hypotheses are later fused by the consensus module.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas import Hypothesis, Incident, TriageResult, UsageStats
from .base import BaseAgent, as_float, as_list, as_str, incident_brief

FAULT_TYPES = [
    "FIBER_CUT", "FIBER_DEGRADATION", "CELL_OUTAGE", "CONGESTION", "MISCONFIG",
    "HARDWARE_FAILURE", "POWER_OUTAGE", "POWER_BROWNOUT", "GPS_SYNC_LOSS",
    "CORE_OVERLOAD", "UPF_DEGRADATION", "DNS_FAILURE", "LICENSE_EXHAUSTION",
    "INTERFERENCE", "OTHER",
]


@dataclass(frozen=True)
class ExpertProfile:
    key: str
    title: str
    scope: str


EXPERTS: list[ExpertProfile] = [
    ExpertProfile(
        key="ran_expert", title="RAN (Radio Access) expert",
        scope="cells, gNodeBs and radio KPIs: cell/RRU outages, radio congestion (PRB), uplink "
        "interference/BLER, and connected-user license exhaustion. Note that radio symptoms can be "
        "caused upstream (transport/power) — follow the dependency chain before blaming the radio.",
    ),
    ExpertProfile(
        key="transport_expert", title="Transport & Infrastructure expert",
        scope="routers, backhaul fiber links, aggregation switches AND site power/rectifiers: fiber "
        "cuts / loss-of-signal, line-card hardware faults / packet loss, and site power failures "
        "(mains loss, battery). A single transport or power fault often explains a whole branch of "
        "downstream RAN alarms.",
    ),
    ExpertProfile(
        key="core_expert", title="Core Network expert",
        scope="AMF/SMF/UPF/DNS: registration failures after a bad config change, compute overload "
        "(CPU exhaustion), and DNS failures that break PDU session setup while radio looks healthy.",
    ),
]

SYSTEM_TMPL = """You are the {title} on an incident bridge for a 5G network.
Your area: {scope}

Access policy (like a real NOC bridge): the alarm board and topology are SHARED, but deep telemetry
(diagnostics, logs, detailed KPI values) is per-team — you can only inspect elements in YOUR domain.
If a tool tells you an element is outside your domain, do not fight it: state in your findings what
you need checked there, and reflect the uncertainty in your confidence.

Method:
1. Look at which elements are anomalous and use the topology to find the *upstream* element that could
   explain them all (avoid blaming a symptom node when a parent is the true cause).
2. Run diagnostics on your prime suspect to CONFIRM — a hypothesis you verified with a diagnostic on
   the blamed element carries far more weight in the bridge vote than an unverified impression.
3. Cross-check the knowledge base / history for a matching signature.

Then give your single best root-cause hypothesis. If the true cause looks OUTSIDE your area, still
report your best guess but set a LOW confidence and say so in the rationale.

Respond with ONLY a JSON object:
{{"faulty_element_id": "the ONE element that is the root cause",
  "fault_type": "one of {fault_types}",
  "root_cause": "one clear sentence",
  "confidence": 0.0-1.0,
  "rationale": "why, referencing the dependency chain",
  "evidence": ["specific tool findings, e.g. 'FIBER-LINK-01 optical_power = -41 dBm (LOS)'"]}}"""

REVISION_TMPL = """You are the {title} on an incident bridge for a 5G network (cross-examination round).
Your area: {scope}

You proposed a hypothesis, but other experts on the bridge disagree. Their hypotheses and the
diagnostics they VERIFIED on their blamed elements are shown below. Weigh their verified evidence
against yours; you may run one or two more tool calls in your own domain to re-check.

Then either DEFEND your hypothesis or REVISE it (you may adopt another expert's element if their
verified evidence better explains the shared alarm picture — upstream causes beat symptom nodes).

Respond with ONLY a JSON object:
{{"faulty_element_id": "the ONE element that is the root cause",
  "fault_type": "one of {fault_types}",
  "root_cause": "one clear sentence",
  "confidence": 0.0-1.0,
  "rationale": "why you defended or revised",
  "evidence": ["specific tool findings"]}}"""


class DiagnosisAgent(BaseAgent):
    tool_names = [
        "query_alarms", "query_kpis", "query_logs", "query_topology",
        "run_diagnostic", "search_knowledge_base", "get_historical_incidents",
    ]

    def __init__(self, llm, ctx, profile: ExpertProfile) -> None:
        super().__init__(llm, ctx)
        self.profile = profile
        self.name = profile.key

    def run(self, incident: Incident, triage: TriageResult, correlation_notes: str, tool_names=None):
        system = SYSTEM_TMPL.format(
            title=self.profile.title, scope=self.profile.scope, fault_types=", ".join(FAULT_TYPES)
        )
        user = (
            incident_brief(incident)
            + f"\n\nTriage: severity={triage.severity.value}, suspected_domain={triage.suspected_domain.value}."
            + f"\nKnowledge correlation:\n{correlation_notes}"
        )
        run = self.invoke(system, user, tool_names=tool_names)
        return self._to_hypothesis(run)

    def revise(self, incident: Incident, own: Hypothesis, rivals_block: str, tool_names=None):
        """Cross-examination round: defend or revise given rivals' verified evidence."""
        system = REVISION_TMPL.format(
            title=self.profile.title, scope=self.profile.scope, fault_types=", ".join(FAULT_TYPES)
        )
        user = (
            incident_brief(incident)
            + f"\n\nYOUR current hypothesis: element={own.faulty_element_id}, type={own.fault_type}, "
            + f"confidence={own.confidence:.2f} — {own.root_cause}"
            + f"\n\nOther experts (with their VERIFIED diagnostics):\n{rivals_block}"
        )
        run = self.invoke(system, user, tool_names=tool_names)
        hyp, trace, usage = self._to_hypothesis(run)
        if hyp.faulty_element_id is None:  # malformed revision -> keep the original
            return own, trace, usage
        return hyp, trace, usage

    def _to_hypothesis(self, run):
        d = run.data
        hyp = Hypothesis(
            proposed_by=self.profile.key,
            root_cause=as_str(d.get("root_cause"), "Unclear"),
            faulty_element_id=as_str(d.get("faulty_element_id")) or None,
            fault_type=as_str(d.get("fault_type")) or None,
            confidence=as_float(d.get("confidence"), 0.4),
            rationale=as_str(d.get("rationale")),
            evidence=as_list(d.get("evidence")),
        )
        return hyp, run.trace, run.usage or UsageStats()

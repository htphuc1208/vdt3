"""Incident scenarios with ground-truth labels (for demos and benchmarking).

Each scenario injects one root-cause fault. The resulting alarm storm becomes the
Incident the agents must analyse; the ground-truth fields are used only by the
evaluation harness, never shown to the agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Domain, Incident
from .simulator import FAULT_LIBRARY, NetworkSimulator


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    description: str
    fault_type: str
    element_id: str
    domain: Domain
    root_cause_keywords: list[str] = field(default_factory=list)
    remediation_sop: str = ""
    remediation_keywords: list[str] = field(default_factory=list)


def _kw(fault_type: str) -> list[str]:
    return FAULT_LIBRARY[fault_type].root_cause_keywords


def _sop(fault_type: str) -> str:
    return FAULT_LIBRARY[fault_type].remediation_sop


def _rkw(fault_type: str) -> list[str]:
    return FAULT_LIBRARY[fault_type].remediation_keywords


SCENARIOS: list[Scenario] = [
    Scenario(
        id="fiber_cut", title="Backhaul outage across SITE-A",
        description="A wave of unreachable-node and LOS alarms appeared across a transport branch and its RAN sites; multiple cells report zero throughput.",
        fault_type="FIBER_CUT", element_id="FIBER-LINK-01", domain=Domain.TRANSPORT,
        root_cause_keywords=_kw("FIBER_CUT"), remediation_sop=_sop("FIBER_CUT"), remediation_keywords=_rkw("FIBER_CUT"),
    ),
    Scenario(
        id="cell_outage", title="Single cell out of service",
        description="One cell dropped to zero RRC success and zero throughput while its sibling cells remain healthy.",
        fault_type="CELL_OUTAGE", element_id="RAN-CELL-03", domain=Domain.RAN,
        root_cause_keywords=_kw("CELL_OUTAGE"), remediation_sop=_sop("CELL_OUTAGE"), remediation_keywords=_rkw("CELL_OUTAGE"),
    ),
    Scenario(
        id="congestion", title="Throughput collapse under high load",
        description="Cells under one gNodeB show ~98% PRB utilization, very low throughput and rising latency during a traffic peak.",
        fault_type="CONGESTION", element_id="RAN-GNB-02", domain=Domain.RAN,
        root_cause_keywords=_kw("CONGESTION"), remediation_sop=_sop("CONGESTION"), remediation_keywords=_rkw("CONGESTION"),
    ),
    Scenario(
        id="amf_misconfig", title="Registration failures after a change",
        description="Device registration success collapsed on the core shortly after a change window; reject-cause spikes in the logs.",
        fault_type="MISCONFIG", element_id="CORE-AMF-01", domain=Domain.CORE,
        root_cause_keywords=_kw("MISCONFIG"), remediation_sop=_sop("MISCONFIG"), remediation_keywords=_rkw("MISCONFIG"),
    ),
    Scenario(
        id="linecard_fault", title="Intermittent packet loss on a branch",
        description="A transport router reports rising CRC errors and a flapping interface; downstream nodes see elevated packet loss and reduced throughput.",
        fault_type="HARDWARE_FAILURE", element_id="TRANSPORT-RTR-02", domain=Domain.TRANSPORT,
        root_cause_keywords=_kw("HARDWARE_FAILURE"), remediation_sop=_sop("HARDWARE_FAILURE"), remediation_keywords=_rkw("HARDWARE_FAILURE"),
    ),
    Scenario(
        id="site_power", title="Site running on battery",
        description="A site rectifier reports AC mains failure and a draining battery; all cells at the site are dropping.",
        fault_type="POWER_OUTAGE", element_id="PWR-RECT-01", domain=Domain.POWER,
        root_cause_keywords=_kw("POWER_OUTAGE"), remediation_sop=_sop("POWER_OUTAGE"), remediation_keywords=_rkw("POWER_OUTAGE"),
    ),
    Scenario(
        id="amf_overload", title="Core NF compute overload",
        description="A core network function shows sustained 99% CPU with degraded registration success and rising latency network-wide.",
        fault_type="CORE_OVERLOAD", element_id="CORE-AMF-01", domain=Domain.CORE,
        root_cause_keywords=_kw("CORE_OVERLOAD"), remediation_sop=_sop("CORE_OVERLOAD"), remediation_keywords=_rkw("CORE_OVERLOAD"),
    ),
    Scenario(
        id="dns_failure", title="Data sessions failing, radio healthy",
        description="Radio KPIs look normal but PDU session setup success collapsed; DNS SERVFAILs appear in the core logs.",
        fault_type="DNS_FAILURE", element_id="CORE-DNS-01", domain=Domain.CORE,
        root_cause_keywords=_kw("DNS_FAILURE"), remediation_sop=_sop("DNS_FAILURE"), remediation_keywords=_rkw("DNS_FAILURE"),
    ),
    Scenario(
        id="license", title="New users rejected on a gNodeB",
        description="A gNodeB rejects new RRC setups with a license-limit cause while existing users are unaffected.",
        fault_type="LICENSE_EXHAUSTION", element_id="RAN-GNB-01", domain=Domain.RAN,
        root_cause_keywords=_kw("LICENSE_EXHAUSTION"), remediation_sop=_sop("LICENSE_EXHAUSTION"), remediation_keywords=_rkw("LICENSE_EXHAUSTION"),
    ),
    Scenario(
        id="interference", title="Degraded cell with high BLER",
        description="A single cell shows high uplink BLER and low throughput while neighbouring cells are fine.",
        fault_type="INTERFERENCE", element_id="RAN-CELL-05", domain=Domain.RAN,
        root_cause_keywords=_kw("INTERFERENCE"), remediation_sop=_sop("INTERFERENCE"), remediation_keywords=_rkw("INTERFERENCE"),
    ),
]

_BY_ID = {s.id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in _BY_ID:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(_BY_ID)}")
    return _BY_ID[scenario_id]


def list_scenario_ids() -> list[str]:
    return [s.id for s in SCENARIOS]


def make_simulator(scenario: Scenario) -> NetworkSimulator:
    sim = NetworkSimulator()
    sim.inject(scenario.element_id, scenario.fault_type)
    return sim


def build_incident(scenario: Scenario, sim: NetworkSimulator) -> Incident:
    alarms = sim.get_alarms()
    affected = sorted({a.element_id for a in alarms})
    return Incident(
        id=f"INC-{scenario.id}",
        title=scenario.title,
        description=scenario.description,
        alarms=alarms,
        affected_elements=affected,
    )

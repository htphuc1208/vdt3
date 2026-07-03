"""Incident scenarios with ground-truth labels (for demos and benchmarking).

Each scenario injects one root-cause fault. The resulting alarm storm becomes the
Incident the agents must analyse; the ground-truth fields are used only by the
evaluation harness, never shown to the agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Alarm, Domain, Incident, Severity
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
    suite: str = "telco_v1"
    stress_tags: tuple[str, ...] = field(default_factory=tuple)
    secondary_faults: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    topology: str = "small"  # "small" (default 19-element) | "large" (~95-element)

    @property
    def acceptable_elements(self) -> tuple[str, ...]:
        """Elements that count as correct localization (primary + real secondary faults)."""
        return (self.element_id, *[e for e, _ in self.secondary_faults])


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

STRESS_TAGS = (
    "rag_required",
    "no_rag_solvable",
    "expert_disagreement",
    "arbiter_required",
    "missing_noisy_telemetry",
    "multi_fault",
    "distractor_alarms",
    "no_exact_sop",
)


def _telco_v2_scenarios() -> list[Scenario]:
    """Generate a larger deterministic suite for paper-grade stress testing.

    The simulator remains synthetic, but the generated suite avoids the ten-case
    ceiling of telco_v1 by varying root-cause elements, wording, and stress tags.
    """
    candidates: dict[str, list[str]] = {
        "FIBER_CUT": ["FIBER-LINK-01"],
        "CELL_OUTAGE": ["RAN-CELL-01", "RAN-CELL-02", "RAN-CELL-03", "RAN-CELL-04", "RAN-CELL-05"],
        "CONGESTION": ["RAN-GNB-01", "RAN-GNB-02", "RAN-GNB-03"],
        "MISCONFIG": ["CORE-AMF-01", "CORE-SMF-01"],
        "HARDWARE_FAILURE": ["TRANSPORT-RTR-01", "TRANSPORT-RTR-02", "CORE-RTR-01"],
        "POWER_OUTAGE": ["PWR-RECT-01"],
        "CORE_OVERLOAD": ["CORE-AMF-01", "CORE-SMF-01", "CORE-UPF-01"],
        "DNS_FAILURE": ["CORE-DNS-01"],
        "LICENSE_EXHAUSTION": ["RAN-GNB-01", "RAN-GNB-02", "RAN-GNB-03"],
        "INTERFERENCE": ["RAN-CELL-01", "RAN-CELL-02", "RAN-CELL-03", "RAN-CELL-04", "RAN-CELL-05"],
    }
    descriptions: dict[str, list[str]] = {
        "FIBER_CUT": [
            "Transport LOS alarms are mixed with downstream RAN reachability loss during a maintenance window.",
            "A branch reports zero throughput after an excavation notice near the backhaul route.",
            "Several dependent cells fail together while the optical link shows loss-of-signal symptoms.",
        ],
        "CELL_OUTAGE": [
            "One cell goes out of service while neighbouring cells on the same gNodeB remain healthy.",
            "A single radio sector drops users and reports an RRU heartbeat fault.",
            "Only one cell has zero throughput and RRC success despite healthy transport KPIs.",
        ],
        "CONGESTION": [
            "A traffic hotspot drives very high PRB usage and degraded per-user throughput.",
            "Multiple cells under one gNodeB slow down during a demand spike.",
            "Admission control begins rejecting bearers as the radio layer saturates.",
        ],
        "MISCONFIG": [
            "Registration failures begin shortly after a core change record is applied.",
            "A config drift alarm appears with reject-cause spikes on a core NF.",
            "The control plane degrades after a change window while transport remains stable.",
        ],
        "HARDWARE_FAILURE": [
            "A transport router shows CRC errors and flapping with intermittent downstream loss.",
            "Packet loss appears on one branch while a router card reports hardware errors.",
            "A line-card-like fault produces rising loss and latency without a complete outage.",
        ],
        "POWER_OUTAGE": [
            "A site rectifier reports mains failure and battery depletion during a regional outage.",
            "Power alarms precede radio elements at the same site dropping service.",
            "The whole site degrades while the power unit reports battery operation.",
        ],
        "CORE_OVERLOAD": [
            "A core NF hits sustained CPU saturation and throttles service requests.",
            "Control-plane latency rises while one core component reports resource exhaustion.",
            "Network-wide success rates degrade with one NF at compute capacity.",
        ],
        "DNS_FAILURE": [
            "Radio KPIs stay healthy but session setup fails with name-resolution errors.",
            "DNS SERVFAIL logs coincide with PDU session establishment failures.",
            "Data sessions fail after resolver errors while access and transport look nominal.",
        ],
        "LICENSE_EXHAUSTION": [
            "New users are rejected by a gNodeB while existing connected users remain stable.",
            "A connected-user entitlement limit is reached during a local event.",
            "RRC setup rejects point to license capacity rather than radio RF quality.",
        ],
        "INTERFERENCE": [
            "One cell shows high uplink BLER and a raised noise floor while neighbours are healthy.",
            "Throughput collapses on a single cell with RF interference indicators.",
            "An external uplink noise source appears to affect one radio sector.",
        ],
    }
    secondary_cycle = [
        ("CORE-DNS-01", "DNS_FAILURE"),
        ("RAN-CELL-05", "INTERFERENCE"),
        ("TRANSPORT-RTR-02", "HARDWARE_FAILURE"),
    ]

    scenarios: list[Scenario] = []
    for fault_idx, fault_type in enumerate(candidates):
        spec = FAULT_LIBRARY[fault_type]
        for variant in range(6):
            element_id = candidates[fault_type][variant % len(candidates[fault_type])]
            tag = STRESS_TAGS[(fault_idx + variant) % len(STRESS_TAGS)]
            tags = [tag]
            secondary: tuple[tuple[str, str], ...] = ()
            if tag == "multi_fault":
                secondary_fault = secondary_cycle[(fault_idx + variant) % len(secondary_cycle)]
                if secondary_fault == (element_id, fault_type):
                    secondary_fault = secondary_cycle[(fault_idx + variant + 1) % len(secondary_cycle)]
                secondary = (secondary_fault,)
            if variant == 5 and "no_exact_sop" not in tags:
                tags.append("no_exact_sop")
            desc = descriptions[fault_type][variant % len(descriptions[fault_type])]
            scenarios.append(
                Scenario(
                    id=f"v2_{fault_type.lower()}_{variant + 1}",
                    title=f"Telco v2 {fault_type.lower().replace('_', ' ')} case {variant + 1}",
                    description=desc,
                    fault_type=fault_type,
                    element_id=element_id,
                    domain=spec.domain,
                    root_cause_keywords=_kw(fault_type),
                    remediation_sop=_sop(fault_type),
                    remediation_keywords=_rkw(fault_type),
                    suite="telco_v2",
                    stress_tags=tuple(tags),
                    secondary_faults=secondary,
                )
            )
    return scenarios


TELCO_V2_SCENARIOS: list[Scenario] = _telco_v2_scenarios()


def _v3(id: str, title: str, description: str, fault_type: str, element_id: str,
        stress_tags: tuple[str, ...] = (), secondary: tuple[tuple[str, str], ...] = ()) -> Scenario:
    spec = FAULT_LIBRARY[fault_type]
    return Scenario(
        id=id, title=title, description=description,
        fault_type=fault_type, element_id=element_id, domain=spec.domain,
        root_cause_keywords=_kw(fault_type), remediation_sop=_sop(fault_type),
        remediation_keywords=_rkw(fault_type),
        suite="telco_v3", stress_tags=stress_tags, secondary_faults=secondary,
        topology="large",
    )


# --------------------------------------------------------------------------- #
# telco_v3 — the "hard" suite. Design principles (literature-grounded):
#  * runs on the ~95-element topology, so exploration cost is real;
#  * cross-domain masquerade faults: the true cause is a QUIET warning in one
#    domain while LOUD misleading symptoms fire in another (the regime where a
#    committee with partitioned expertise can beat one investigator);
#  * multi-fault and false-alarm cases;
#  * descriptions are symptom-only — they never leak the true cause.
# --------------------------------------------------------------------------- #
TELCO_V3_SCENARIOS: list[Scenario] = [
    _v3("v3_fiber_degradation_a", "Widespread poor radio quality in region A",
        "Many cells across several SITE-A/C sites report high BLER and degraded throughput; field teams suspect an interference source.",
        "FIBER_DEGRADATION", "FIBER-LINK-01"),
    _v3("v3_fiber_degradation_b", "Cluster of degraded cells in region B",
        "A cluster of cells under one aggregation branch shows elevated error rates and slow data; radio quality complaints are rising.",
        "FIBER_DEGRADATION", "FIBER-LINK-02"),
    _v3("v3_power_brownout_e", "Sporadic restarts across one site",
        "Elements at one site keep warm-restarting with watchdog resets; users see intermittent service. No total outage.",
        "POWER_BROWNOUT", "PWR-RECT-05"),
    _v3("v3_power_brownout_a", "Flapping nodes at SITE-A",
        "Several SITE-A elements flap in and out of reach; symptoms look like unstable hardware or transport.",
        "POWER_BROWNOUT", "PWR-RECT-01"),
    _v3("v3_gps_sync_d1", "Interference-like degradation on co-sited cells",
        "Multiple cells at one site show raised uplink noise and BLER, as if an external interferer appeared.",
        "GPS_SYNC_LOSS", "RAN-GNB-D1"),
    _v3("v3_gps_sync_g2", "Uplink noise across a site in region B",
        "Co-sited cells degrade together with interference-pattern symptoms; no single cell stands out.",
        "GPS_SYNC_LOSS", "RAN-GNB-G2"),
    _v3("v3_upf_degradation", "Network-wide slow data, radio looks loaded",
        "Users at EVERY site report high latency and slow downloads; radio and transport teams each suspect congestion on their side.",
        "UPF_DEGRADATION", "CORE-UPF-01"),
    _v3("v3_backbone_card", "Massive multi-region packet loss",
        "Both regions report intermittent packet loss and degraded throughput; dozens of elements raise alarms simultaneously.",
        "HARDWARE_FAILURE", "CORE-RTR-01"),
    _v3("v3_multifault_fiber_intf", "Region-B outage plus an unrelated degraded cell",
        "A whole region-B branch is unreachable, and separately one cell in another area shows poor uplink quality.",
        "FIBER_CUT", "FIBER-LINK-03",
        secondary=(("RAN-CELL-C1-2", "INTERFERENCE"),)),
    _v3("v3_multifault_power_dns", "Site outage while data sessions also fail elsewhere",
        "One site is dropping on battery, while independently new data sessions fail across the network with resolution errors.",
        "POWER_OUTAGE", "PWR-RECT-04",
        secondary=(("CORE-DNS-01", "DNS_FAILURE"),)),
    _v3("v3_license_false_alarm", "User rejections plus a suspicious core alarm",
        "New users are rejected at one gNodeB; at the same time a transient config-mismatch alarm fires on a core NF, drawing attention.",
        "LICENSE_EXHAUSTION", "RAN-GNB-F2", stress_tags=("distractor_alarms",)),
    _v3("v3_cell_outage_noise", "One dead cell amid unrelated alarm noise",
        "A single cell is out of service while unrelated transient alarms fire elsewhere in the network.",
        "CELL_OUTAGE", "RAN-CELL-H1-2", stress_tags=("distractor_alarms",)),
]

_BY_ID = {s.id: s for s in [*SCENARIOS, *TELCO_V2_SCENARIOS, *TELCO_V3_SCENARIOS]}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in _BY_ID:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(_BY_ID)}")
    return _BY_ID[scenario_id]


def list_scenario_ids(suite: str = "telco_v1") -> list[str]:
    if suite == "all":
        return [s.id for s in [*SCENARIOS, *TELCO_V2_SCENARIOS, *TELCO_V3_SCENARIOS]]
    if suite == "telco_v2":
        return [s.id for s in TELCO_V2_SCENARIOS]
    if suite == "telco_v3":
        return [s.id for s in TELCO_V3_SCENARIOS]
    return [s.id for s in SCENARIOS]


def make_simulator(scenario: Scenario) -> NetworkSimulator:
    from .topology import build_large_topology

    topo = build_large_topology() if scenario.topology == "large" else None
    sim = NetworkSimulator(topo)
    sim.inject(scenario.element_id, scenario.fault_type)
    for element_id, fault_type in scenario.secondary_faults:
        sim.inject(element_id, fault_type)
    return sim


def build_incident(scenario: Scenario, sim: NetworkSimulator) -> Incident:
    alarms = sim.get_alarms()
    if "missing_noisy_telemetry" in scenario.stress_tags and len(alarms) > 1:
        alarms = [a for a in alarms if a.element_id == scenario.element_id] or alarms[:1]
    if "distractor_alarms" in scenario.stress_tags:
        alarms = [
            *alarms,
            Alarm(
                element_id="CORE-SMF-01",
                severity=Severity.MINOR,
                name="TRANSIENT_SESSION_RETRY",
                probable_cause="Background retry storm unrelated to the primary incident",
            ),
        ]
    affected = sorted({a.element_id for a in alarms})
    return Incident(
        id=f"INC-{scenario.id}",
        title=scenario.title,
        description=scenario.description,
        alarms=alarms,
        affected_elements=affected,
    )

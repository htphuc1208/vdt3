"""Network simulator: healthy telemetry + fault injection + remediation.

This is the *environment* the agents observe through tools. It is deterministic
(stable telemetry for a given fault) so runs are reproducible, but it is only a
data source — all reasoning is done live by the LLM agents.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from ..schemas import Alarm, Domain, KPISample, LogEntry, NetworkElement, Severity
from .topology import Topology, build_default_topology

# --------------------------------------------------------------------------- #
# Metric metadata: healthy baseline, normal operating range, unit, noise.
# --------------------------------------------------------------------------- #
METRIC_META: dict[str, dict] = {
    "rrc_setup_success_rate": {"healthy": 99.4, "normal": (98.0, 100.0), "unit": "%", "noise": 0.4},
    "prb_utilization": {"healthy": 45.0, "normal": (0.0, 85.0), "unit": "%", "noise": 6.0},
    "dl_throughput_mbps": {"healthy": 120.0, "normal": (40.0, 2000.0), "unit": "Mbps", "noise": 12.0},
    "ul_bler": {"healthy": 2.0, "normal": (0.0, 10.0), "unit": "%", "noise": 0.8},
    "latency_ms": {"healthy": 12.0, "normal": (0.0, 30.0), "unit": "ms", "noise": 2.0},
    "packet_loss": {"healthy": 0.05, "normal": (0.0, 0.5), "unit": "%", "noise": 0.05},
    "cpu_utilization": {"healthy": 38.0, "normal": (0.0, 80.0), "unit": "%", "noise": 6.0},
    "registration_success_rate": {"healthy": 99.8, "normal": (99.0, 100.0), "unit": "%", "noise": 0.15},
    "session_setup_success_rate": {"healthy": 99.5, "normal": (98.0, 100.0), "unit": "%", "noise": 0.3},
    "optical_rx_power": {"healthy": -8.0, "normal": (-22.0, -4.0), "unit": "dBm", "noise": 0.5},
    "battery_level": {"healthy": 100.0, "normal": (50.0, 100.0), "unit": "%", "noise": 0.0},
    "connected_users": {"healthy": 250.0, "normal": (0.0, 850.0), "unit": "users", "noise": 30.0},
    "license_usage": {"healthy": 62.0, "normal": (0.0, 95.0), "unit": "%", "noise": 5.0},
}

# Which metrics are relevant to which element type.
from ..schemas import ElementType  # noqa: E402

ELEMENT_METRICS: dict[ElementType, list[str]] = {
    ElementType.CELL: ["rrc_setup_success_rate", "prb_utilization", "dl_throughput_mbps", "ul_bler", "latency_ms"],
    ElementType.GNB: ["cpu_utilization", "connected_users", "license_usage"],
    ElementType.AGG_SWITCH: ["packet_loss", "latency_ms"],
    ElementType.ROUTER: ["packet_loss", "latency_ms", "optical_rx_power"],
    ElementType.FIBER_LINK: ["optical_rx_power", "packet_loss"],
    ElementType.CORE_NF: ["cpu_utilization", "registration_success_rate", "session_setup_success_rate", "latency_ms"],
    ElementType.POWER_UNIT: ["battery_level"],
}


# --------------------------------------------------------------------------- #
# Fault specification
# --------------------------------------------------------------------------- #
@dataclass
class FaultSpec:
    fault_type: str
    domain: Domain
    root_cause_label: str
    root_cause_keywords: list[str]
    remediation_sop: str
    remediation_keywords: list[str]
    # (scope, severity, alarm_name, probable_cause)
    alarm_specs: list[tuple[str, Severity, str, str]] = field(default_factory=list)
    # (scope, metric, absolute_value)
    kpi_overrides: list[tuple[str, str, float]] = field(default_factory=list)
    # (scope, level, message_template)  -> may use {id}
    log_specs: list[tuple[str, str, str]] = field(default_factory=list)
    # diagnostic check name -> result template (may use {id})
    diagnostics: dict[str, str] = field(default_factory=dict)
    summary: str = ""


# fault_type -> FaultSpec
FAULT_LIBRARY: dict[str, FaultSpec] = {
    "FIBER_CUT": FaultSpec(
        fault_type="FIBER_CUT", domain=Domain.TRANSPORT,
        root_cause_label="Fiber cut on the backhaul transport link (loss of signal)",
        root_cause_keywords=["fiber", "transport", "link", "los", "optical", "cut"],
        remediation_sop="SOP-TRANSPORT-FIBER",
        remediation_keywords=["fiber", "splice", "repair", "reroute", "field team", "optical", "restore link"],
        alarm_specs=[
            ("self", Severity.CRITICAL, "LOS", "Loss of optical signal / fiber break"),
            ("downstream", Severity.MAJOR, "NODE_UNREACHABLE", "Upstream transport path down"),
        ],
        kpi_overrides=[
            ("self", "optical_rx_power", -41.0),
            ("self", "packet_loss", 100.0),
            ("downstream", "packet_loss", 100.0),
            ("downstream", "dl_throughput_mbps", 0.0),
            ("downstream", "rrc_setup_success_rate", 0.0),
            ("downstream", "latency_ms", 999.0),
        ],
        log_specs=[
            ("self", "CRITICAL", "{id}: optical Rx power -41 dBm, below LOS threshold; interface DOWN"),
            ("downstream", "MAJOR", "{id}: heartbeat to upstream lost, node unreachable"),
        ],
        diagnostics={
            "interface_status": "{id}: interface DOWN (LOS)",
            "optical_power": "{id}: Rx = -41.0 dBm (LOS, threshold -30 dBm)",
        },
        summary="Backhaul fiber break isolates a whole transport branch.",
    ),
    "CELL_OUTAGE": FaultSpec(
        fault_type="CELL_OUTAGE", domain=Domain.RAN,
        root_cause_label="RRU/cell hardware failure — cell out of service",
        root_cause_keywords=["cell", "rru", "hardware", "outage", "out of service"],
        remediation_sop="SOP-RAN-CELL-RESET",
        remediation_keywords=["restart", "reset", "rru", "replace", "cell", "reboot"],
        alarm_specs=[("self", Severity.CRITICAL, "CELL_DOWN", "Cell unavailable / RRU heartbeat lost")],
        kpi_overrides=[
            ("self", "rrc_setup_success_rate", 0.0),
            ("self", "dl_throughput_mbps", 0.0),
            ("self", "prb_utilization", 0.0),
        ],
        log_specs=[("self", "CRITICAL", "{id}: cell OUT_OF_SERVICE, RRU heartbeat lost")],
        diagnostics={"cell_status": "{id}: OUT_OF_SERVICE (RRU fault)"},
        summary="A single cell/RRU has failed; neighbours are healthy.",
    ),
    "CONGESTION": FaultSpec(
        fault_type="CONGESTION", domain=Domain.RAN,
        root_cause_label="Radio congestion / capacity exhaustion on the gNodeB",
        root_cause_keywords=["congestion", "prb", "load", "capacity", "utilization"],
        remediation_sop="SOP-RAN-CONGESTION",
        remediation_keywords=["load balanc", "carrier", "capacity", "offload", "re-home", "add cell"],
        alarm_specs=[
            ("self", Severity.MINOR, "HIGH_LOAD", "Sustained PRB utilization > 95%"),
            ("downstream", Severity.WARNING, "DEGRADED_QOS", "QoS degraded under high load"),
        ],
        kpi_overrides=[
            ("downstream", "prb_utilization", 98.0),
            ("downstream", "dl_throughput_mbps", 8.0),
            ("downstream", "latency_ms", 78.0),
            ("downstream", "rrc_setup_success_rate", 92.0),
        ],
        log_specs=[("self", "MINOR", "{id}: PRB utilization sustained ~98%, admission control rejecting bearers")],
        diagnostics={"load": "{id}: PRB util 98%, 900+ active users, admission control active"},
        summary="A gNodeB is capacity-bound; throughput collapses under load.",
    ),
    "MISCONFIG": FaultSpec(
        fault_type="MISCONFIG", domain=Domain.CORE,
        root_cause_label="Misconfiguration after a change — registrations rejected",
        root_cause_keywords=["config", "misconfig", "change", "registration", "rollback"],
        remediation_sop="SOP-CORE-CONFIG-ROLLBACK",
        remediation_keywords=["rollback", "revert", "config", "change", "golden"],
        alarm_specs=[("self", Severity.MAJOR, "CONFIG_MISMATCH", "Config drift vs golden after recent change")],
        kpi_overrides=[
            ("self", "registration_success_rate", 12.0),
            ("self", "latency_ms", 42.0),
        ],
        log_specs=[("self", "MAJOR", "{id}: registration reject cause #11 spike after change window CR-4471")],
        diagnostics={"config_audit": "{id}: PLMN list mismatch vs golden config (changed 20 min ago, CR-4471)"},
        summary="A bad config change broke device registration on a core NF.",
    ),
    "HARDWARE_FAILURE": FaultSpec(
        fault_type="HARDWARE_FAILURE", domain=Domain.TRANSPORT,
        root_cause_label="Router line-card hardware fault (intermittent packet loss)",
        root_cause_keywords=["hardware", "line card", "router", "packet loss", "crc"],
        remediation_sop="SOP-TRANSPORT-CARD-REPLACE",
        remediation_keywords=["replace", "line card", "hardware", "swap", "rma"],
        alarm_specs=[
            ("self", Severity.MAJOR, "CARD_FAULT", "Line card CRC errors / flapping"),
            ("downstream", Severity.MINOR, "PACKET_LOSS", "Elevated packet loss on the path"),
        ],
        kpi_overrides=[
            ("self", "packet_loss", 7.5),
            ("self", "latency_ms", 62.0),
            ("downstream", "packet_loss", 7.0),
            ("downstream", "dl_throughput_mbps", 22.0),
        ],
        log_specs=[("self", "MAJOR", "{id}: line card 2 CRC errors rising, interface flapping, temp high")],
        diagnostics={"hardware": "{id}: line card 2 CRC errors + high temperature (failing)"},
        summary="A failing line card injects intermittent loss into a transport branch.",
    ),
    "POWER_OUTAGE": FaultSpec(
        fault_type="POWER_OUTAGE", domain=Domain.POWER,
        root_cause_label="Site power failure — rectifier on depleting battery",
        root_cause_keywords=["power", "mains", "battery", "site", "rectifier"],
        remediation_sop="SOP-POWER-RESTORE",
        remediation_keywords=["generator", "power", "mains", "rectifier", "fuel", "dispatch", "restore power"],
        alarm_specs=[
            ("self", Severity.CRITICAL, "MAINS_FAIL", "AC mains failure — running on battery"),
            ("site", Severity.MAJOR, "SITE_ON_BATTERY", "Site elements on depleting battery"),
        ],
        kpi_overrides=[
            ("self", "battery_level", 24.0),
            ("site", "rrc_setup_success_rate", 0.0),
            ("site", "dl_throughput_mbps", 0.0),
        ],
        log_specs=[
            ("self", "CRITICAL", "{id}: AC mains lost, on battery 24% and discharging"),
            ("site", "MAJOR", "{id}: powered by battery, imminent outage"),
        ],
        diagnostics={"power": "{id}: mains FAIL, battery 24% discharging (~35 min remaining)"},
        summary="Loss of mains power takes a whole site offline as the battery drains.",
    ),
    "CORE_OVERLOAD": FaultSpec(
        fault_type="CORE_OVERLOAD", domain=Domain.CORE,
        root_cause_label="Core NF compute overload (CPU exhaustion)",
        root_cause_keywords=["overload", "cpu", "capacity", "scale", "resource"],
        remediation_sop="SOP-CORE-SCALEOUT",
        remediation_keywords=["scale", "add instance", "capacity", "cpu", "horizontal", "autoscal"],
        alarm_specs=[("self", Severity.MAJOR, "HIGH_CPU", "Sustained CPU > 95%, request throttling")],
        kpi_overrides=[
            ("self", "cpu_utilization", 99.0),
            ("self", "registration_success_rate", 78.0),
            ("self", "latency_ms", 58.0),
        ],
        log_specs=[("self", "MAJOR", "{id}: CPU 99%, request queue overflow, throttling registrations")],
        diagnostics={"resource": "{id}: CPU 99%, memory 88%, thread pool saturated"},
        summary="A core NF is compute-bound; registrations degrade network-wide.",
    ),
    "DNS_FAILURE": FaultSpec(
        fault_type="DNS_FAILURE", domain=Domain.CORE,
        root_cause_label="Core DNS failure — PDU session establishment failing",
        root_cause_keywords=["dns", "resolution", "session", "servfail", "resolver"],
        remediation_sop="SOP-CORE-DNS-FAILOVER",
        remediation_keywords=["dns", "restart", "failover", "resolver", "secondary"],
        alarm_specs=[("self", Severity.MAJOR, "DNS_UNRESOLVED", "APN/SRV DNS resolution failing")],
        kpi_overrides=[
            ("self", "session_setup_success_rate", 22.0),
            ("self", "latency_ms", 48.0),
        ],
        log_specs=[("self", "MAJOR", "{id}: APN DNS queries returning SERVFAIL; PDU session setup failing")],
        diagnostics={"dns": "{id}: nslookup apn.internal -> SERVFAIL; upstream resolver unreachable"},
        summary="DNS is down: radio is fine but data sessions cannot be set up.",
    ),
    "LICENSE_EXHAUSTION": FaultSpec(
        fault_type="LICENSE_EXHAUSTION", domain=Domain.RAN,
        root_cause_label="License exhaustion on the gNodeB (connected-user cap reached)",
        root_cause_keywords=["license", "limit", "users", "entitlement", "cap"],
        remediation_sop="SOP-RAN-LICENSE",
        remediation_keywords=["license", "increase", "entitlement", "upgrade", "capacity"],
        alarm_specs=[("self", Severity.MINOR, "LICENSE_LIMIT", "Connected-user license exhausted")],
        kpi_overrides=[
            ("self", "license_usage", 100.0),
            ("self", "connected_users", 900.0),
            ("downstream", "rrc_setup_success_rate", 85.0),
        ],
        log_specs=[("self", "MINOR", "{id}: RRC setup reject — license limit reached (max connected users)")],
        diagnostics={"license": "{id}: connected-user license 100% used (900/900)"},
        summary="New users are rejected because the gNodeB user license is maxed out.",
    ),
    "INTERFERENCE": FaultSpec(
        fault_type="INTERFERENCE", domain=Domain.RAN,
        root_cause_label="External uplink interference on a cell (high BLER)",
        root_cause_keywords=["interference", "bler", "rf", "uplink", "noise"],
        remediation_sop="SOP-RAN-INTERFERENCE",
        remediation_keywords=["interference", "spectrum", "rf", "hunt", "pim", "scan"],
        alarm_specs=[("self", Severity.MINOR, "HIGH_INTERFERENCE", "Elevated uplink noise / BLER")],
        kpi_overrides=[
            ("self", "ul_bler", 28.0),
            ("self", "dl_throughput_mbps", 15.0),
            ("self", "latency_ms", 40.0),
        ],
        log_specs=[("self", "MINOR", "{id}: uplink RSSI elevated (-95 dBm noise floor), BLER 28%")],
        diagnostics={"rf": "{id}: UL interference, noise floor -95 dBm, external source suspected"},
        summary="An external RF source raises the noise floor on one cell.",
    ),
}


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
@dataclass
class ActiveFault:
    element_id: str
    spec: FaultSpec


def _stable_noise(element_id: str, metric: str) -> float:
    h = hashlib.sha256(f"{element_id}:{metric}".encode()).digest()
    frac = int.from_bytes(h[:4], "big") / 0xFFFFFFFF  # [0,1)
    return (frac - 0.5) * 2.0  # [-1, 1]


class NetworkSimulator:
    """Holds network state, injects faults, and serves telemetry to the tools."""

    def __init__(self, topology: Optional[Topology] = None) -> None:
        self.topology = topology or build_default_topology()
        self.active: list[ActiveFault] = []

    # -- fault lifecycle -----------------------------------------------------
    def inject(self, element_id: str, fault_type: str) -> None:
        spec = FAULT_LIBRARY.get(fault_type)
        if spec is None:
            raise ValueError(f"Unknown fault type: {fault_type}")
        if element_id not in self.topology:
            raise ValueError(f"Unknown element: {element_id}")
        self.active.append(ActiveFault(element_id, spec))

    def clear(self, element_id: str | None = None) -> None:
        if element_id is None:
            self.active.clear()
        else:
            self.active = [f for f in self.active if f.element_id != element_id]

    def is_healthy(self) -> bool:
        return not self.active

    # -- scope resolution ----------------------------------------------------
    def _scope_ids(self, fault: ActiveFault, scope: str) -> list[str]:
        el = self.topology.get(fault.element_id)
        if el is None:
            return []
        if scope == "self":
            return [el.id]
        if scope == "downstream":
            return [d.id for d in self.topology.descendants(el.id)]
        if scope == "site":
            return [e.id for e in self.topology.at_site(el.site) if e.id != el.id]
        if scope == "parent" and el.parent_id:
            return [el.parent_id]
        return []

    def _overrides_for(self, element_id: str) -> dict[str, float]:
        """Merge all active-fault KPI overrides that apply to ``element_id``."""
        out: dict[str, float] = {}
        for fault in self.active:
            for scope, metric, value in fault.spec.kpi_overrides:
                if element_id in self._scope_ids(fault, scope):
                    out[metric] = value
        return out

    # -- telemetry: KPIs -----------------------------------------------------
    def _kpi_for(self, element: NetworkElement, metric: str, overrides: dict[str, float]) -> KPISample:
        meta = METRIC_META[metric]
        lo, hi = meta["normal"]
        if metric in overrides:
            value = overrides[metric]
        else:
            value = meta["healthy"] + _stable_noise(element.id, metric) * meta["noise"]
        value = round(value, 2)
        anomalous = value < lo or value > hi
        return KPISample(
            element_id=element.id, metric=metric, value=value, unit=meta["unit"],
            normal_range=(lo, hi), is_anomalous=anomalous,
        )

    def get_kpis(self, element_id: str | None = None, metric: str | None = None) -> list[KPISample]:
        samples: list[KPISample] = []
        if element_id:
            elements = [self.topology.get(element_id)] if element_id in self.topology else []
        else:
            elements = self.topology.all()
        for el in elements:
            if el is None:
                continue
            overrides = self._overrides_for(el.id)
            metrics = ELEMENT_METRICS.get(el.type, [])
            if metric:
                metrics = [metric] if metric in metrics else []
            for m in metrics:
                sample = self._kpi_for(el, m, overrides)
                # when scanning the whole network, only surface anomalies (bounded output)
                if element_id or sample.is_anomalous:
                    samples.append(sample)
        return samples

    # -- telemetry: alarms ---------------------------------------------------
    def get_alarms(self, element_id: str | None = None, severity: str | None = None) -> list[Alarm]:
        alarms: list[Alarm] = []
        seen: set[tuple[str, str]] = set()
        for fault in self.active:
            for scope, sev, name, cause in fault.spec.alarm_specs:
                for eid in self._scope_ids(fault, scope):
                    key = (eid, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    alarms.append(Alarm(element_id=eid, severity=sev, name=name, probable_cause=cause))
        if element_id:
            alarms = [a for a in alarms if a.element_id == element_id]
        if severity:
            alarms = [a for a in alarms if a.severity.value == severity.upper()]
        # CRITICAL first
        order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.WARNING: 3}
        return sorted(alarms, key=lambda a: order[a.severity])

    # -- telemetry: logs -----------------------------------------------------
    def get_logs(self, element_id: str | None = None, level: str | None = None, limit: int = 20) -> list[LogEntry]:
        logs: list[LogEntry] = []
        for fault in self.active:
            for scope, lvl, template in fault.spec.log_specs:
                for eid in self._scope_ids(fault, scope):
                    logs.append(LogEntry(element_id=eid, level=lvl, message=template.format(id=eid)))
        if element_id:
            logs = [l for l in logs if l.element_id == element_id]
        if level:
            logs = [l for l in logs if l.level.upper() == level.upper()]
        return logs[:limit]

    # -- diagnostics ---------------------------------------------------------
    def run_diagnostic(self, element_id: str, check: str) -> str:
        for fault in self.active:
            if element_id in self._scope_ids(fault, "self") or element_id == fault.element_id:
                if check in fault.spec.diagnostics:
                    return fault.spec.diagnostics[check].format(id=element_id)
        el = self.topology.get(element_id)
        if el is None:
            return f"Unknown element {element_id}"
        return f"{element_id}: {check} check nominal (no fault detected on this element)."

    # -- remediation ---------------------------------------------------------
    def apply_remediation(self, action: str, element_id: str | None = None) -> dict:
        """Attempt a fix. Succeeds only when it targets the real fault correctly."""
        action_l = (action or "").lower()
        for fault in list(self.active):
            faulty = fault.element_id
            spec = fault.spec
            keyword_hit = any(k in action_l for k in spec.remediation_keywords) or spec.remediation_sop.lower() in action_l
            target_ok = (
                element_id is None
                or element_id == faulty
                or faulty.lower() in action_l
            )
            if keyword_hit and target_ok:
                self.clear(faulty)
                return {
                    "status": "resolved",
                    "element_id": faulty,
                    "message": f"Remediation applied to {faulty}; KPIs recovered to normal.",
                }
        if self.active:
            return {
                "status": "no_effect",
                "message": "Action applied but KPIs did NOT recover — wrong target or wrong fix; root cause persists.",
            }
        return {"status": "already_healthy", "message": "Network is already healthy."}

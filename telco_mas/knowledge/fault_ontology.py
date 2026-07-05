"""Public fault ontology used consistently by MAS and the strong baseline.

The ontology separates canonical root-fault families from alarm-condition names.
Only high-specificity root conditions may seed a deterministic candidate; generic
propagated symptoms such as HIGH_INTERFERENCE and DEGRADED_QOS deliberately do not.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..schemas import Alarm, Domain, NetworkElement


CANONICAL_FAULT_TYPES = (
    "FIBER_CUT",
    "FIBER_DEGRADATION",
    "CELL_OUTAGE",
    "CONGESTION",
    "MISCONFIG",
    "HARDWARE_FAILURE",
    "POWER_OUTAGE",
    "POWER_BROWNOUT",
    "GPS_SYNC_LOSS",
    "CORE_OVERLOAD",
    "UPF_DEGRADATION",
    "DNS_FAILURE",
    "LICENSE_EXHAUSTION",
    "INTERFERENCE",
)

FAULT_DOMAINS: dict[str, Domain] = {
    "FIBER_CUT": Domain.TRANSPORT,
    "FIBER_DEGRADATION": Domain.TRANSPORT,
    "CELL_OUTAGE": Domain.RAN,
    "CONGESTION": Domain.RAN,
    "MISCONFIG": Domain.CORE,
    "HARDWARE_FAILURE": Domain.TRANSPORT,
    "POWER_OUTAGE": Domain.POWER,
    "POWER_BROWNOUT": Domain.POWER,
    "GPS_SYNC_LOSS": Domain.RAN,
    "CORE_OVERLOAD": Domain.CORE,
    "UPF_DEGRADATION": Domain.CORE,
    "DNS_FAILURE": Domain.CORE,
    "LICENSE_EXHAUSTION": Domain.RAN,
    "INTERFERENCE": Domain.RAN,
}

# (alarm name, element type) -> canonical root family. These are intentionally
# high-specificity conditions. Generic propagated alarms are excluded.
ROOT_CONDITION_SIGNATURES: dict[tuple[str, str], str] = {
    ("LOS", "FIBER_LINK"): "FIBER_CUT",
    ("OPTICAL_DEGRADED", "FIBER_LINK"): "FIBER_DEGRADATION",
    ("CELL_DOWN", "CELL"): "CELL_OUTAGE",
    ("HIGH_LOAD", "GNB"): "CONGESTION",
    ("CONFIG_MISMATCH", "CORE_NF"): "MISCONFIG",
    ("CARD_FAULT", "ROUTER"): "HARDWARE_FAILURE",
    ("MAINS_FAIL", "POWER_UNIT"): "POWER_OUTAGE",
    ("VOLTAGE_SAG", "POWER_UNIT"): "POWER_BROWNOUT",
    ("SYNC_HOLDOVER", "GNB"): "GPS_SYNC_LOSS",
    ("HIGH_CPU", "CORE_NF"): "CORE_OVERLOAD",
    ("UP_LATENCY_DEGRADED", "CORE_NF"): "UPF_DEGRADATION",
    ("DNS_UNRESOLVED", "CORE_NF"): "DNS_FAILURE",
    ("LICENSE_LIMIT", "GNB"): "LICENSE_EXHAUSTION",
}

FAULT_EXPLANATIONS: dict[str, str] = {
    "FIBER_CUT": "Fiber loss of signal indicates a cut transport link.",
    "FIBER_DEGRADATION": "Degraded optical power indicates a lossy backhaul fiber path.",
    "CELL_OUTAGE": "The cell or RRU is out of service after losing its heartbeat.",
    "CONGESTION": "Radio load and PRB capacity are exhausted at the gNodeB.",
    "MISCONFIG": "A core configuration mismatch after a change is rejecting service requests.",
    "HARDWARE_FAILURE": "A transport router card hardware fault is causing CRC errors and flapping.",
    "POWER_OUTAGE": "Loss of mains power is draining the site rectifier battery.",
    "POWER_BROWNOUT": "A site rectifier voltage sag is causing shared-device restarts.",
    "GPS_SYNC_LOSS": "GNSS synchronization loss and holdover drift are causing TDD timing interference.",
    "CORE_OVERLOAD": "Core network-function CPU overload is throttling requests.",
    "UPF_DEGRADATION": "UPF user-plane packet-processing degradation is raising network-wide latency.",
    "DNS_FAILURE": "Core DNS resolution failure is preventing PDU session establishment.",
    "LICENSE_EXHAUSTION": "The gNodeB connected-user license entitlement is exhausted.",
    "INTERFERENCE": "External radio interference is raising the uplink noise floor and BLER.",
}

REMEDIATION_FAMILY_HINTS: dict[str, str] = {
    "FIBER_CUT": "repair/splice or reroute the failed optical link",
    "FIBER_DEGRADATION": "clean, reseat, re-terminate, or replace the degraded optical path",
    "CELL_OUTAGE": "restart the cell/RRU, then replace failed radio hardware if needed",
    "CONGESTION": "offload traffic, load-balance, or add radio capacity",
    "MISCONFIG": "roll back the offending core configuration to the validated baseline",
    "HARDWARE_FAILURE": "replace or fail over the faulty router/line card",
    "POWER_OUTAGE": "restore mains/generator power and stabilize the rectifier",
    "POWER_BROWNOUT": "stabilize voltage or repair/replace the regulator/rectifier",
    "GPS_SYNC_LOSS": "restore GNSS/PTP timing, replace the timing source, and resynchronize",
    "CORE_OVERLOAD": "scale out the overloaded core network function",
    "UPF_DEGRADATION": "drain traffic and restart or scale out the UPF user-plane instance",
    "DNS_FAILURE": "fail over or restart the core resolver and restore name resolution",
    "LICENSE_EXHAUSTION": "increase the connected-user entitlement or redistribute users",
    "INTERFERENCE": "locate and remove the RF/PIM interferer using spectrum analysis",
}

_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "FIBER_CUT": ("fiber cut", "loss of signal", "optical los"),
    "FIBER_DEGRADATION": ("fiber degradation", "optical degradation", "degraded optical"),
    "CELL_OUTAGE": ("cell outage", "cell down", "out of service", "rru failure"),
    "CONGESTION": ("radio congestion", "high load", "prb exhaustion"),
    "MISCONFIG": ("misconfig", "configuration mismatch", "config mismatch"),
    "HARDWARE_FAILURE": ("hardware failure", "card fault", "line card"),
    "POWER_OUTAGE": ("power outage", "mains fail", "mains failure"),
    "POWER_BROWNOUT": ("power brownout", "voltage sag", "brownout"),
    "GPS_SYNC_LOSS": ("gps sync loss", "gnss loss", "sync holdover", "timing loss"),
    "CORE_OVERLOAD": ("core overload", "cpu overload", "high cpu"),
    "UPF_DEGRADATION": ("upf degradation", "up latency degraded", "user plane degradation"),
    "DNS_FAILURE": ("dns failure", "dns unresolved", "resolver failure"),
    "LICENSE_EXHAUSTION": ("license exhaustion", "license limit", "license cap"),
    "INTERFERENCE": ("radio interference", "high interference", "rf interference"),
}


def root_condition_family(alarm: Alarm, element: NetworkElement | None) -> str | None:
    if element is None:
        return None
    return ROOT_CONDITION_SIGNATURES.get((alarm.name.upper(), element.type.value))


def direct_root_condition_families(
    element: NetworkElement | None,
    alarms: Iterable[Alarm],
) -> set[str]:
    if element is None:
        return set()
    return {
        family
        for alarm in alarms
        if alarm.element_id == element.id
        for family in [root_condition_family(alarm, element)]
        if family
    }


def canonicalize_fault_type(
    predicted: str | None,
    *,
    element: NetworkElement | None,
    alarms: Iterable[Alarm],
    root_cause: str = "",
) -> str | None:
    """Map alarm-condition wording to a canonical root family.

    A unique direct, element-type-compatible root condition has priority. This
    is observable evidence, not an evaluator label. Text normalization is the
    fallback and is shared by both compared systems.
    """
    direct = direct_root_condition_families(element, alarms)
    if len(direct) == 1:
        return next(iter(direct))

    raw = (predicted or "").strip().upper()
    if raw in CANONICAL_FAULT_TYPES:
        return raw
    text = _normalize_text(" ".join(part for part in (predicted or "", root_cause) if part))
    matches = [
        family
        for family, aliases in _TEXT_ALIASES.items()
        if any(alias in text for alias in aliases)
    ]
    return matches[0] if len(matches) == 1 else (raw or None)


def domain_compatible(fault_type: str | None, element: NetworkElement | None) -> bool | None:
    if not fault_type or element is None:
        return None
    expected = FAULT_DOMAINS.get(fault_type)
    return None if expected is None else element.domain == expected


def earliest_specific_root_candidates(incident, topology) -> list[tuple[str, str, Alarm]]:
    """Return high-specificity root conditions at the first observed event time."""
    timestamped = [alarm for alarm in incident.alarms if alarm.raised_at is not None]
    if not timestamped:
        return []
    earliest = min(alarm.raised_at for alarm in timestamped)
    candidates: dict[tuple[str, str], Alarm] = {}
    for alarm in timestamped:
        if alarm.raised_at != earliest:
            continue
        family = root_condition_family(alarm, topology.get(alarm.element_id))
        if family:
            candidates[(alarm.element_id, family)] = alarm
    return [
        (element_id, family, alarm)
        for (element_id, family), alarm in sorted(candidates.items())
    ]


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

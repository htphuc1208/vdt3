"""Seeded, outcome-unseen synthetic telecom confirmatory suite.

The generator is intentionally independent from model outputs. Each fault
family receives the same four nuisance profiles, while root placement is
shuffled by a declared seed. Generate the artifact once after freezing the
algorithm, then evaluate that exact artifact by hash.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from typing import Any

from ..environment.scenarios import Scenario
from ..environment.simulator import FAULT_LIBRARY


DESIGN_VERSION = "telco-v4.0"
VARIANTS_PER_FAMILY = 4

NUISANCE_PROFILES: dict[str, tuple[str, ...]] = {
    "complete": (),
    "alarm_distractor": ("distractor_alarms",),
    "incomplete_alarm_view": ("missing_noisy_telemetry",),
    "kb_holdout": ("no_exact_sop",),
}


def _large_gnbs() -> list[str]:
    return [
        "RAN-GNB-01",
        "RAN-GNB-02",
        "RAN-GNB-03",
        *[f"RAN-GNB-{site}{index}" for site in "CDEFGH" for index in (1, 2)],
    ]


def _large_cells() -> list[str]:
    return [
        "RAN-CELL-01",
        "RAN-CELL-02",
        "RAN-CELL-03",
        "RAN-CELL-04",
        "RAN-CELL-05",
        *[
            f"RAN-CELL-{site}{gnb}-{cell}"
            for site in "CDEFGH"
            for gnb in (1, 2)
            for cell in (1, 2, 3)
        ],
    ]


ROOT_CANDIDATES: dict[str, list[str]] = {
    "FIBER_CUT": ["FIBER-LINK-01", "FIBER-LINK-02", "FIBER-LINK-03"],
    "CELL_OUTAGE": _large_cells(),
    "CONGESTION": _large_gnbs(),
    "MISCONFIG": ["CORE-AMF-01", "CORE-SMF-01", "CORE-UDM-01", "CORE-PCF-01"],
    "HARDWARE_FAILURE": [
        "CORE-RTR-01",
        "TRANSPORT-RTR-01",
        "TRANSPORT-RTR-02",
        "TRANSPORT-RTR-03",
        "TRANSPORT-RTR-04",
    ],
    "POWER_OUTAGE": ["PWR-RECT-01", *[f"PWR-RECT-{index:02d}" for index in range(3, 9)]],
    "CORE_OVERLOAD": ["CORE-AMF-01", "CORE-SMF-01", "CORE-UPF-01", "CORE-UDM-01", "CORE-PCF-01"],
    "DNS_FAILURE": ["CORE-DNS-01"],
    "LICENSE_EXHAUSTION": _large_gnbs(),
    "INTERFERENCE": _large_cells(),
    "FIBER_DEGRADATION": ["FIBER-LINK-01", "FIBER-LINK-02", "FIBER-LINK-03"],
    "POWER_BROWNOUT": ["PWR-RECT-01", *[f"PWR-RECT-{index:02d}" for index in range(3, 9)]],
    "GPS_SYNC_LOSS": _large_gnbs(),
    "UPF_DEGRADATION": ["CORE-UPF-01"],
}


# Symptom-only descriptions. They do not name the evaluator fault label or root ID.
SYMPTOM_DESCRIPTIONS: dict[str, tuple[str, str, str, str]] = {
    "FIBER_CUT": (
        "A transport branch and all dependent radio nodes became unreachable within minutes.",
        "Several co-dependent sites lost data service together while other branches remained healthy.",
        "A regional group of cells dropped to zero throughput after abrupt path alarms.",
        "One aggregation subtree vanished from monitoring and downstream sessions stopped.",
    ),
    "CELL_OUTAGE": (
        "One sector stopped carrying users while sibling sectors at the site remained normal.",
        "A single radio sector reports zero setup success and no traffic; nearby coverage is available.",
        "Only one cell is out of service, with no matching degradation on its parent or neighbors.",
        "Users attached to one sector were dropped while the rest of the site stayed operational.",
    ),
    "CONGESTION": (
        "Busy-hour demand caused several sectors under one node to lose per-user throughput.",
        "Admission failures rise with load while radio availability and transport reachability stay intact.",
        "Latency and bearer rejects increase only during a sustained local traffic peak.",
        "A radio node serves many active users but quality falls as resource use approaches its limit.",
    ),
    "MISCONFIG": (
        "Control-plane success fell shortly after an approved change, without a transport anomaly.",
        "A core function began rejecting valid requests after configuration synchronization completed.",
        "Service failures align with a recent policy update and a drift warning on one network function.",
        "A change window ended, then one control-plane component returned systematic reject causes.",
    ),
    "HARDWARE_FAILURE": (
        "Intermittent loss spreads below one routing component while interfaces repeatedly flap.",
        "A network branch has bursty packet loss, rising CRC counters, and no complete path outage.",
        "Multiple dependent sites degrade whenever one forwarding component reports thermal errors.",
        "Traffic recovers and fails in short cycles around a shared transport device.",
    ),
    "POWER_OUTAGE": (
        "Every network element at one site is losing service as its remaining reserve declines.",
        "A whole site became unavailable together, while upstream transport stayed reachable.",
        "Co-located radio and transport equipment shut down after facility alarms appeared.",
        "One location is approaching total service loss and all local equipment shows the same dependency.",
    ),
    "CORE_OVERLOAD": (
        "A control-plane function has long queues and network-wide request success is falling.",
        "Service latency rises across sites while one core component remains at sustained high utilization.",
        "Registrations are throttled globally even though access and transport capacity look normal.",
        "A core service cannot keep up with demand and begins rejecting work under a growing queue.",
    ),
    "DNS_FAILURE": (
        "Radio access is healthy, but new data sessions fail when service names are resolved.",
        "Session establishment collapses alongside repeated resolver errors in the packet core.",
        "Devices register successfully but cannot create data sessions that require internal name lookup.",
        "A network-wide data setup failure coincides with unsuccessful APN-related queries.",
    ),
    "LICENSE_EXHAUSTION": (
        "Existing users remain connected while new access attempts are rejected at one radio node.",
        "One node stops accepting additional subscribers despite normal RF and compute measurements.",
        "Connection setup fails only after the active-user count reaches a fixed ceiling.",
        "A local event triggers deterministic admission rejects without packet loss or poor signal.",
    ),
    "INTERFERENCE": (
        "One sector has a raised noise floor and high uplink block errors while neighbors are clean.",
        "Radio quality collapses on a single cell without matching load or transport symptoms.",
        "A localized RF anomaly raises retransmissions but leaves co-sited sectors mostly unaffected.",
        "One coverage sector develops persistent uplink errors and reduced user throughput.",
    ),
    "FIBER_DEGRADATION": (
        "Many downstream cells look RF-impaired, but their degradation follows one shared path.",
        "A branch shows modest packet errors while dependent radio sectors report much louder quality alarms.",
        "Several sites degrade together with high retransmissions, despite unrelated radio conditions.",
        "User quality fluctuates across one subtree and follows a weak common transport symptom.",
    ),
    "POWER_BROWNOUT": (
        "Co-located devices restart intermittently even though no component reports a permanent failure.",
        "One site's equipment flaps during load peaks, with independent hardware faults suspected.",
        "Warm restarts occur across radio and transport devices sharing the same location.",
        "Packet loss and setup failures appear in bursts whenever several site devices reset together.",
    ),
    "GPS_SYNC_LOSS": (
        "Co-sited TDD sectors develop slot-aligned uplink noise rather than a localized RF source.",
        "Several cells at one site show interference-like symptoms at the same time.",
        "Radio errors spread across sectors sharing one timing source while adjacent sites remain stable.",
        "A site's uplink quality worsens after its synchronization state enters extended holdover.",
    ),
    "UPF_DEGRADATION": (
        "Users across every access region see slow data while registration remains available.",
        "Network-wide packet latency rises without a common radio or backhaul bottleneck.",
        "All sites report poor user-plane quality while one shared packet-processing path queues traffic.",
        "Download performance falls globally even though access load differs widely by site.",
    ),
}


def build_holdout_scenarios(seed: int) -> list[Scenario]:
    """Construct a balanced 56-case suite without consulting model outcomes."""
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for family in sorted(ROOT_CANDIDATES):
        candidates = list(ROOT_CANDIDATES[family])
        rng.shuffle(candidates)
        profiles = list(NUISANCE_PROFILES)
        rng.shuffle(profiles)
        spec = FAULT_LIBRARY[family]
        for variant in range(VARIANTS_PER_FAMILY):
            profile = profiles[variant]
            token = hashlib.sha256(
                f"{DESIGN_VERSION}:{seed}:{family}:{variant}".encode("utf-8")
            ).hexdigest()[:12]
            scenarios.append(
                Scenario(
                    id=f"v4_{token}",
                    title=f"Network service incident {token[:6]}",
                    description=SYMPTOM_DESCRIPTIONS[family][variant],
                    fault_type=family,
                    element_id=candidates[variant % len(candidates)],
                    domain=spec.domain,
                    root_cause_keywords=list(spec.root_cause_keywords),
                    remediation_sop=spec.remediation_sop,
                    remediation_keywords=list(spec.remediation_keywords),
                    suite="telco_v4",
                    stress_tags=NUISANCE_PROFILES[profile],
                    topology="large",
                )
            )
    rng.shuffle(scenarios)
    return scenarios


def design_manifest(
    seed: int,
    scenarios: list[Scenario] | None = None,
    *,
    seed_source: str | None = None,
) -> dict[str, Any]:
    selected = scenarios or build_holdout_scenarios(seed)
    return {
        "design_version": DESIGN_VERSION,
        "seed": seed,
        "seed_source": seed_source,
        "case_count": len(selected),
        "variants_per_fault_family": VARIANTS_PER_FAMILY,
        "fault_family_counts": dict(sorted(Counter(item.fault_type for item in selected).items())),
        "nuisance_profile_counts": {
            name: sum(item.stress_tags == tags for item in selected)
            for name, tags in NUISANCE_PROFILES.items()
        },
        "construction": [
            "balanced crossed design: every fault family receives every nuisance profile once",
            "root placement shuffled from topology-valid candidates by the declared seed",
            "opaque case IDs and symptom-only descriptions",
            "no multi-root cases in the confirmatory primary estimand",
            "generator does not consume model predictions or benchmark outcomes",
        ],
    }

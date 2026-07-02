"""Standard Operating Procedures (playbooks) and a historical incident base.

The SOP ids line up with the remediation SOPs referenced by the fault library, so
retrieving the right SOP is equivalent to matching the correct remediation.
"""
from __future__ import annotations

SOP_LIBRARY: list[dict] = [
    {
        "id": "SOP-TRANSPORT-FIBER",
        "title": "Backhaul fiber break / Loss of Signal",
        "domain": "TRANSPORT",
        "symptoms": "Critical LOS alarm on a fiber link, optical Rx power below -30 dBm, whole downstream branch unreachable, cells report zero throughput.",
        "steps": [
            "Confirm LOS: read optical Rx power and interface status on the fiber link.",
            "Check whether a protection/alternate path exists and reroute traffic if possible.",
            "Raise a field-team dispatch to locate and splice/repair the fiber break.",
            "After repair, verify optical power and interface come back UP and downstream KPIs recover.",
        ],
    },
    {
        "id": "SOP-RAN-CELL-RESET",
        "title": "Single cell / RRU out of service",
        "domain": "RAN",
        "symptoms": "One cell reports CELL_DOWN, zero RRC success and zero throughput; neighbouring cells healthy.",
        "steps": [
            "Confirm the cell/RRU status via diagnostics.",
            "Soft-restart the affected cell/RRU remotely.",
            "If the restart fails, schedule an RRU hardware replacement.",
            "Verify RRC success and throughput recover on the cell.",
        ],
    },
    {
        "id": "SOP-RAN-CONGESTION",
        "title": "Radio congestion / capacity exhaustion",
        "domain": "RAN",
        "symptoms": "Sustained ~98% PRB utilization, very low per-user throughput, rising latency, admission-control rejects during peak load.",
        "steps": [
            "Confirm sustained high PRB utilization and active-user count.",
            "Apply load balancing / traffic offload to neighbour cells or carriers.",
            "Activate an additional carrier or cell to add capacity if available.",
            "Verify PRB utilization drops and throughput recovers.",
        ],
    },
    {
        "id": "SOP-CORE-CONFIG-ROLLBACK",
        "title": "Registration failures after a config change",
        "domain": "CORE",
        "symptoms": "Registration success collapses on a core NF right after a change window; config-mismatch alarm and reject-cause spikes.",
        "steps": [
            "Identify the recent change (change record) on the core NF.",
            "Audit running config vs golden config to locate the drift.",
            "Roll back / revert the offending configuration change.",
            "Verify registration success recovers to normal.",
        ],
    },
    {
        "id": "SOP-TRANSPORT-CARD-REPLACE",
        "title": "Router line-card hardware fault",
        "domain": "TRANSPORT",
        "symptoms": "Rising CRC errors, flapping interface and high temperature on a router line card; intermittent packet loss downstream.",
        "steps": [
            "Confirm the failing line card via hardware diagnostics (CRC/temperature).",
            "Shift traffic off the affected card/port if redundancy allows.",
            "Replace / RMA the faulty line card.",
            "Verify packet loss clears and throughput recovers.",
        ],
    },
    {
        "id": "SOP-POWER-RESTORE",
        "title": "Site power failure (mains lost, on battery)",
        "domain": "POWER",
        "symptoms": "MAINS_FAIL alarm, rectifier on battery with a dropping charge level, whole site at risk of outage.",
        "steps": [
            "Confirm mains status and remaining battery level on the rectifier.",
            "Dispatch a field team with a generator / restore mains power.",
            "Prioritise sites by remaining battery runtime.",
            "Verify mains restored and site KPIs recover.",
        ],
    },
    {
        "id": "SOP-CORE-SCALEOUT",
        "title": "Core NF compute overload",
        "domain": "CORE",
        "symptoms": "Sustained CPU > 95% on a core NF, request throttling, degraded registration/session success.",
        "steps": [
            "Confirm CPU/memory saturation on the NF.",
            "Scale out horizontally (add an instance) or increase allocated capacity.",
            "Shed / throttle non-critical load temporarily if needed.",
            "Verify CPU drops and success rates recover.",
        ],
    },
    {
        "id": "SOP-CORE-DNS-FAILOVER",
        "title": "Core DNS resolution failure",
        "domain": "CORE",
        "symptoms": "APN/SRV DNS queries return SERVFAIL, PDU session setup fails while radio KPIs are normal.",
        "steps": [
            "Confirm DNS failures (nslookup SERVFAIL) on the core DNS.",
            "Fail over to the secondary DNS / restart the DNS service.",
            "Confirm upstream resolver reachability.",
            "Verify session setup success recovers.",
        ],
    },
    {
        "id": "SOP-RAN-LICENSE",
        "title": "gNodeB connected-user license exhausted",
        "domain": "RAN",
        "symptoms": "RRC setups rejected with a license-limit cause, connected-user license at 100%, existing users unaffected.",
        "steps": [
            "Confirm the license usage on the gNodeB.",
            "Increase / extend the connected-user license entitlement.",
            "Rebalance users to neighbour cells as a temporary measure.",
            "Verify RRC setup success recovers.",
        ],
    },
    {
        "id": "SOP-RAN-INTERFERENCE",
        "title": "External uplink interference on a cell",
        "domain": "RAN",
        "symptoms": "High uplink BLER, elevated noise floor and low throughput on a single cell; neighbours fine.",
        "steps": [
            "Confirm elevated uplink noise floor / BLER on the cell.",
            "Run an interference hunt / spectrum scan to locate the source.",
            "Mitigate (remove source, retune, or adjust UL power control).",
            "Verify BLER and throughput recover.",
        ],
    },
]

HISTORICAL_INCIDENTS: list[dict] = [
    {
        "id": "HIST-2024-0142", "fault_type": "FIBER_CUT", "domain": "TRANSPORT",
        "symptoms": "LOS on a backhaul fiber, downstream sites unreachable, cells at zero throughput.",
        "root_cause": "Construction dug through the backhaul fiber.",
        "resolution": "Field team spliced the fiber; traffic rerouted meanwhile (SOP-TRANSPORT-FIBER).",
    },
    {
        "id": "HIST-2024-0210", "fault_type": "CELL_OUTAGE", "domain": "RAN",
        "symptoms": "A single cell CELL_DOWN, zero RRC, neighbours healthy.",
        "root_cause": "RRU power-amplifier hardware failure.",
        "resolution": "Remote restart failed, RRU replaced (SOP-RAN-CELL-RESET).",
    },
    {
        "id": "HIST-2024-0333", "fault_type": "CONGESTION", "domain": "RAN",
        "symptoms": "98% PRB utilization, throughput collapse at rush hour on a busy gNodeB.",
        "root_cause": "Insufficient capacity for a traffic hotspot.",
        "resolution": "Load balancing + extra carrier activated (SOP-RAN-CONGESTION).",
    },
    {
        "id": "HIST-2024-0401", "fault_type": "MISCONFIG", "domain": "CORE",
        "symptoms": "Registration success dropped after a change window, reject cause #11 spike.",
        "root_cause": "Bad PLMN list pushed in change CR-4471.",
        "resolution": "Config rolled back to golden (SOP-CORE-CONFIG-ROLLBACK).",
    },
    {
        "id": "HIST-2024-0455", "fault_type": "HARDWARE_FAILURE", "domain": "TRANSPORT",
        "symptoms": "Rising CRC errors and flapping on a router card, intermittent downstream loss.",
        "root_cause": "Degrading line card.",
        "resolution": "Line card replaced (SOP-TRANSPORT-CARD-REPLACE).",
    },
    {
        "id": "HIST-2024-0501", "fault_type": "POWER_OUTAGE", "domain": "POWER",
        "symptoms": "MAINS_FAIL, rectifier on battery, whole site dropping.",
        "root_cause": "Regional power cut, generator not started.",
        "resolution": "Generator dispatched, mains restored (SOP-POWER-RESTORE).",
    },
    {
        "id": "HIST-2024-0588", "fault_type": "CORE_OVERLOAD", "domain": "CORE",
        "symptoms": "Core NF at 99% CPU, registration throttled network-wide.",
        "root_cause": "Traffic growth exceeded provisioned compute.",
        "resolution": "Scaled out the NF horizontally (SOP-CORE-SCALEOUT).",
    },
    {
        "id": "HIST-2024-0620", "fault_type": "DNS_FAILURE", "domain": "CORE",
        "symptoms": "Radio healthy but PDU sessions failing, DNS SERVFAIL in logs.",
        "root_cause": "Primary DNS resolver crashed.",
        "resolution": "Failed over to secondary DNS and restarted service (SOP-CORE-DNS-FAILOVER).",
    },
    {
        "id": "HIST-2024-0705", "fault_type": "LICENSE_EXHAUSTION", "domain": "RAN",
        "symptoms": "New RRC setups rejected with license-limit cause on one gNodeB.",
        "root_cause": "Connected-user license cap reached during an event.",
        "resolution": "License entitlement increased (SOP-RAN-LICENSE).",
    },
    {
        "id": "HIST-2024-0777", "fault_type": "INTERFERENCE", "domain": "RAN",
        "symptoms": "Single cell high UL BLER and low throughput, elevated noise floor.",
        "root_cause": "External illegal repeater raised the uplink noise floor.",
        "resolution": "Interference hunt located and removed the source (SOP-RAN-INTERFERENCE).",
    },
]

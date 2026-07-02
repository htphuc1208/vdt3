"""A small but realistic 5G network topology.

The parent chain models the physical/logical dependency path
(Core → Transport → Aggregation → gNodeB → Cell), which is what makes a single
root-cause fault light up alarms across many downstream elements — the core
challenge the multi-agent system must solve (localisation under alarm storms).
"""
from __future__ import annotations

from ..schemas import Domain, ElementType, NetworkElement

_ELEMENTS: list[NetworkElement] = [
    # --- Core network functions ---
    NetworkElement(id="CORE-AMF-01", name="AMF (Access & Mobility)", type=ElementType.CORE_NF, domain=Domain.CORE),
    NetworkElement(id="CORE-SMF-01", name="SMF (Session Mgmt)", type=ElementType.CORE_NF, domain=Domain.CORE),
    NetworkElement(id="CORE-UPF-01", name="UPF (User Plane)", type=ElementType.CORE_NF, domain=Domain.CORE),
    NetworkElement(id="CORE-DNS-01", name="Core DNS", type=ElementType.CORE_NF, domain=Domain.CORE),
    # --- Transport ---
    NetworkElement(id="CORE-RTR-01", name="Core Gateway Router", type=ElementType.ROUTER, domain=Domain.TRANSPORT),
    NetworkElement(id="FIBER-LINK-01", name="Backhaul Fiber A", type=ElementType.FIBER_LINK, domain=Domain.TRANSPORT, parent_id="CORE-RTR-01"),
    NetworkElement(id="TRANSPORT-RTR-01", name="Aggregation Router A", type=ElementType.ROUTER, domain=Domain.TRANSPORT, parent_id="FIBER-LINK-01"),
    NetworkElement(id="TRANSPORT-RTR-02", name="Aggregation Router B", type=ElementType.ROUTER, domain=Domain.TRANSPORT, parent_id="CORE-RTR-01"),
    NetworkElement(id="AGG-SW-01", name="Access Switch A", type=ElementType.AGG_SWITCH, domain=Domain.TRANSPORT, parent_id="TRANSPORT-RTR-01", site="SITE-A"),
    NetworkElement(id="AGG-SW-02", name="Access Switch B", type=ElementType.AGG_SWITCH, domain=Domain.TRANSPORT, parent_id="TRANSPORT-RTR-02", site="SITE-B"),
    # --- RAN (gNodeBs) ---
    NetworkElement(id="RAN-GNB-01", name="gNodeB A1", type=ElementType.GNB, domain=Domain.RAN, parent_id="AGG-SW-01", site="SITE-A"),
    NetworkElement(id="RAN-GNB-02", name="gNodeB B1", type=ElementType.GNB, domain=Domain.RAN, parent_id="AGG-SW-02", site="SITE-B"),
    NetworkElement(id="RAN-GNB-03", name="gNodeB A2", type=ElementType.GNB, domain=Domain.RAN, parent_id="AGG-SW-01", site="SITE-A"),
    # --- RAN (cells) ---
    NetworkElement(id="RAN-CELL-01", name="Cell A1-1", type=ElementType.CELL, domain=Domain.RAN, parent_id="RAN-GNB-01", site="SITE-A"),
    NetworkElement(id="RAN-CELL-02", name="Cell A1-2", type=ElementType.CELL, domain=Domain.RAN, parent_id="RAN-GNB-01", site="SITE-A"),
    NetworkElement(id="RAN-CELL-03", name="Cell B1-1", type=ElementType.CELL, domain=Domain.RAN, parent_id="RAN-GNB-02", site="SITE-B"),
    NetworkElement(id="RAN-CELL-04", name="Cell B1-2", type=ElementType.CELL, domain=Domain.RAN, parent_id="RAN-GNB-02", site="SITE-B"),
    NetworkElement(id="RAN-CELL-05", name="Cell A2-1", type=ElementType.CELL, domain=Domain.RAN, parent_id="RAN-GNB-03", site="SITE-A"),
    # --- Power ---
    NetworkElement(id="PWR-RECT-01", name="Rectifier SITE-A", type=ElementType.POWER_UNIT, domain=Domain.POWER, site="SITE-A"),
]


class Topology:
    """Read-only view of the network graph with dependency helpers."""

    def __init__(self, elements: list[NetworkElement] | None = None) -> None:
        self.elements: dict[str, NetworkElement] = {e.id: e for e in (elements or _ELEMENTS)}

    def __contains__(self, element_id: str) -> bool:
        return element_id in self.elements

    def get(self, element_id: str) -> NetworkElement | None:
        return self.elements.get(element_id)

    def all(self) -> list[NetworkElement]:
        return list(self.elements.values())

    def by_domain(self, domain: Domain) -> list[NetworkElement]:
        return [e for e in self.elements.values() if e.domain == domain]

    def children(self, element_id: str) -> list[NetworkElement]:
        return [e for e in self.elements.values() if e.parent_id == element_id]

    def descendants(self, element_id: str) -> list[NetworkElement]:
        """All elements below ``element_id`` in the dependency tree (excl. self)."""
        out: list[NetworkElement] = []
        stack = [element_id]
        while stack:
            current = stack.pop()
            for child in self.children(current):
                out.append(child)
                stack.append(child.id)
        return out

    def at_site(self, site: str | None) -> list[NetworkElement]:
        if not site:
            return []
        return [e for e in self.elements.values() if e.site == site]


def build_default_topology() -> Topology:
    return Topology()

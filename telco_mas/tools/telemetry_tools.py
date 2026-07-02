"""Read-only telemetry tools (TAMO-style multimodal observation).

Each function takes the SessionContext (duck-typed as ``ctx`` with ``.sim``) and
returns a compact JSON string for the LLM to read.
"""
from __future__ import annotations

import json
from typing import Any


def query_topology(ctx: Any, element_id: str | None = None, domain: str | None = None) -> str:
    topo = ctx.sim.topology
    if element_id:
        el = topo.get(element_id)
        if el is None:
            return json.dumps({"error": f"unknown element {element_id}"})
        chain = []
        cur = el.parent_id
        while cur:
            parent = topo.get(cur)
            if not parent:
                break
            chain.append(parent.id)
            cur = parent.parent_id
        return json.dumps(
            {
                "element": el.model_dump(),
                "parents_upstream": chain,
                "children": [c.id for c in topo.children(el.id)],
                "descendants": [d.id for d in topo.descendants(el.id)],
            },
            default=str,
        )
    elements = topo.all()
    if domain:
        elements = [e for e in elements if e.domain.value == domain.upper()]
    return json.dumps(
        [
            {"id": e.id, "name": e.name, "type": e.type.value, "domain": e.domain.value, "parent_id": e.parent_id, "site": e.site}
            for e in elements
        ],
        default=str,
    )


def query_alarms(ctx: Any, element_id: str | None = None, severity: str | None = None) -> str:
    alarms = ctx.sim.get_alarms(element_id=element_id, severity=severity)
    return json.dumps([a.model_dump() for a in alarms], default=str)


def query_kpis(ctx: Any, element_id: str | None = None, metric: str | None = None) -> str:
    samples = ctx.sim.get_kpis(element_id=element_id, metric=metric)
    if not samples:
        return json.dumps({"note": "no KPI anomalies found for that query", "samples": []})
    return json.dumps([s.model_dump() for s in samples], default=str)


def query_logs(ctx: Any, element_id: str | None = None, level: str | None = None, limit: int = 20) -> str:
    logs = ctx.sim.get_logs(element_id=element_id, level=level, limit=limit)
    return json.dumps([l.model_dump() for l in logs], default=str)

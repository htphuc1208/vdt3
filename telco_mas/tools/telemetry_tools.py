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
    if element_id is None:
        return json.dumps(_global_kpi_summary(ctx, samples), default=str)
    out = []
    for s in samples:
        if ctx.can_inspect(s.element_id):
            out.append(s.model_dump())
        else:
            # Out-of-domain elements: coarse view only (shared NOC board shows
            # WHERE anomalies are; detailed counters belong to the domain team).
            out.append({"element_id": s.element_id, "metric": s.metric,
                        "is_anomalous": s.is_anomalous,
                        "note": "detail restricted — outside your domain"})
    return json.dumps(out, default=str)


def _global_kpi_summary(ctx: Any, samples: list[Any]) -> dict[str, Any]:
    """Return a non-oracular network KPI scan.

    A real NOC can see that anomaly volume is high by domain/site/metric, but a
    global scan should not hand every suspect element and raw value to a single
    agent. Detailed values require an explicit element drill-down.
    """

    by_domain: dict[str, int] = {}
    by_site: dict[str, int] = {}
    by_metric: dict[str, int] = {}
    for sample in samples:
        element = ctx.sim.topology.get(sample.element_id)
        domain = element.domain.value if element else "UNKNOWN"
        site = element.site if element and element.site else "UNKNOWN"
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_site[site] = by_site.get(site, 0) + 1
        by_metric[sample.metric] = by_metric.get(sample.metric, 0) + 1
    return {
        "scope": "global_anomaly_summary",
        "anomaly_count": len(samples),
        "by_domain": dict(sorted(by_domain.items())),
        "by_site": dict(sorted(by_site.items())),
        "by_metric": dict(sorted(by_metric.items())),
        "detail_policy": "Global KPI scan hides element IDs and raw readings; call query_kpis with element_id for drill-down.",
    }


def query_logs(ctx: Any, element_id: str | None = None, level: str | None = None, limit: int = 20) -> str:
    if element_id and not ctx.can_inspect(element_id):
        return json.dumps({"error": f"{element_id} is outside your domain — its logs belong to "
                                    "another expert team. Report your findings to the bridge instead."})
    logs = ctx.sim.get_logs(element_id=element_id, level=level, limit=limit)
    logs = [l for l in logs if ctx.can_inspect(l.element_id)]
    return json.dumps([l.model_dump() for l in logs], default=str)

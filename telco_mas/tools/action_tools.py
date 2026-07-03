"""Action tools: run diagnostics and (simulated) apply remediation."""
from __future__ import annotations

import json
from typing import Any


def run_diagnostic(ctx: Any, element_id: str, check: str | None = None) -> str:
    sim = ctx.sim
    if element_id not in sim.topology:
        return json.dumps({"error": f"unknown element {element_id}"})
    if not ctx.can_inspect(element_id):
        return json.dumps({"error": f"{element_id} is outside your domain — deep diagnostics on it "
                                    "belong to another expert team. State what you need in your findings."})
    if check:
        return json.dumps({"element_id": element_id, "check": check, "result": sim.run_diagnostic(element_id, check)})
    # No specific check: run every diagnostic that any active fault exposes for this element.
    results: dict[str, str] = {}
    for fault in sim.active:
        if element_id == fault.element_id or element_id in sim._scope_ids(fault, "self"):
            for name in fault.spec.diagnostics:
                results[name] = sim.run_diagnostic(element_id, name)
    if not results:
        results["status"] = f"{element_id}: all standard checks nominal (no fault on this element)."
    return json.dumps({"element_id": element_id, "diagnostics": results})


def apply_remediation(ctx: Any, action: str, element_id: str | None = None) -> str:
    result = ctx.sim.apply_remediation(action=action, element_id=element_id)
    return json.dumps(result)

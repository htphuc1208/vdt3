"""Tool registry: dispatch returns well-formed JSON and actions affect the sim."""
import json

from telco_mas.environment.scenarios import get_scenario, make_simulator
from telco_mas.tools.registry import SessionContext, dispatch, openai_specs


def _ctx(scenario_id="fiber_cut"):
    return SessionContext.create(make_simulator(get_scenario(scenario_id)))


def test_specs_expose_all_tools():
    names = {t["function"]["name"] for t in openai_specs()}
    assert {"query_alarms", "query_kpis", "query_topology", "search_knowledge_base",
            "run_diagnostic", "apply_remediation"} <= names


def test_telemetry_tools():
    ctx = _ctx()
    alarms = json.loads(dispatch(ctx, "query_alarms", {}))
    assert isinstance(alarms, list) and alarms
    topo = json.loads(dispatch(ctx, "query_topology", {"element_id": "FIBER-LINK-01"}))
    assert "descendants" in topo and topo["descendants"]
    kpis = json.loads(dispatch(ctx, "query_kpis", {"element_id": "FIBER-LINK-01"}))
    assert isinstance(kpis, list)


def test_kb_tool_retrieves_correct_sop():
    ctx = _ctx()
    hits = json.loads(dispatch(ctx, "search_knowledge_base", {"query": "fiber optical loss of signal link down"}))
    assert hits and hits[0]["sop_id"] == "SOP-TRANSPORT-FIBER"


def test_remediation_resolves_only_when_correct():
    ctx = _ctx()
    wrong = json.loads(dispatch(ctx, "apply_remediation", {"action": "reboot", "element_id": "CORE-DNS-01"}))
    assert wrong["status"] == "no_effect"
    right = json.loads(dispatch(ctx, "apply_remediation",
                                {"action": "dispatch field team to repair fiber", "element_id": "FIBER-LINK-01"}))
    assert right["status"] == "resolved"
    assert ctx.sim.is_healthy()

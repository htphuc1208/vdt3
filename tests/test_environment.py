"""The simulated environment behaves consistently for every scenario."""
import pytest

from telco_mas.environment.scenarios import SCENARIOS, build_incident, make_simulator
from telco_mas.environment.simulator import FAULT_LIBRARY


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_symptoms_and_remediation(scenario):
    sim = make_simulator(scenario)

    # 1. faults produce alarms and KPI anomalies
    alarms = sim.get_alarms()
    assert alarms, f"{scenario.id}: expected alarms"
    anomalies = [k for k in sim.get_kpis() if k.is_anomalous]
    assert anomalies, f"{scenario.id}: expected KPI anomalies"

    incident = build_incident(scenario, sim)
    assert incident.alarms and incident.affected_elements

    # 2. a diagnostic on the faulty element reveals the fault
    diag = sim.run_diagnostic(scenario.element_id, list(FAULT_LIBRARY[scenario.fault_type].diagnostics)[0])
    assert "nominal" not in diag.lower()

    # 3. a wrong remediation does NOT resolve it
    bad = sim.apply_remediation("please investigate", element_id=scenario.element_id)
    assert bad["status"] == "no_effect"
    assert not sim.is_healthy()

    # 4. the correct remediation resolves it
    spec = FAULT_LIBRARY[scenario.fault_type]
    action = f"{spec.remediation_keywords[0]} on {scenario.element_id}"
    good = sim.apply_remediation(action, element_id=scenario.element_id)
    assert good["status"] == "resolved"
    assert sim.is_healthy()


def test_fiber_cut_blast_radius():
    """A fiber cut should light up downstream RAN elements, not just the link."""
    scenario = next(s for s in SCENARIOS if s.id == "fiber_cut")
    sim = make_simulator(scenario)
    affected = {a.element_id for a in sim.get_alarms()}
    assert "FIBER-LINK-01" in affected
    assert any(e.startswith("RAN-CELL") for e in affected)

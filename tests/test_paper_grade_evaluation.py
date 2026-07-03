import json

from telco_mas.environment.scenarios import (
    STRESS_TAGS,
    build_incident,
    get_scenario,
    list_scenario_ids,
    make_simulator,
)
from telco_mas.evaluation.metrics import score_result
from telco_mas.evaluation.stats import aggregate_ci, paired_bootstrap_effect, paired_mcnemar
from telco_mas.llm import LLMClient
from telco_mas.pipeline import run
from telco_mas.schemas import (
    ConsensusResult,
    PipelineResult,
    RemediationPlan,
    UsageStats,
    ValidationResult,
)
from telco_mas.tools.registry import SessionContext, dispatch


def test_strict_diagnosis_requires_fault_family_and_causal_explanation():
    scenario = get_scenario("fiber_cut")
    incident = build_incident(scenario, make_simulator(scenario))
    result = PipelineResult(
        incident=incident,
        consensus=ConsensusResult(
            root_cause="fiber cut upstream",
            faulty_element_id=scenario.element_id,
            fault_type="FIBER_CUT",
        ),
        remediation=RemediationPlan(sop_id=scenario.remediation_sop),
        remediation_action="dispatch team to repair fiber FIBER-LINK-01",
        remediation_target_element_id=scenario.element_id,
        validation=ValidationResult(resolved=True, recovered_kpis=["dl_throughput@RAN-CELL-01"]),
        usage=UsageStats(total_tokens=1000),
    )
    score = score_result(result, scenario)
    assert score["root_cause_correct"] is True
    assert score["causal_explanation_correct"] is True
    assert score["diagnosis_correct"] is True
    assert score["end_to_end_correct"] is True

    result.consensus.fault_type = "DNS_FAILURE"
    score = score_result(result, scenario)
    assert score["root_cause_correct"] is True
    assert score["diagnosis_loose_correct"] is True
    assert score["diagnosis_correct"] is False


def test_remediation_submetrics_classify_unresolved_correct_diagnosis():
    scenario = get_scenario("dns_failure")
    incident = build_incident(scenario, make_simulator(scenario))
    result = PipelineResult(
        incident=incident,
        consensus=ConsensusResult(
            root_cause="dns resolver servfail blocks session setup",
            faulty_element_id=scenario.element_id,
            fault_type=scenario.fault_type,
        ),
        remediation=RemediationPlan(sop_id="SOP-RAN-CONGESTION"),
        remediation_action="load balance radio users on RAN-GNB-02",
        remediation_target_element_id="RAN-GNB-02",
        validation=ValidationResult(resolved=False),
        usage=UsageStats(total_tokens=1000),
    )
    score = score_result(result, scenario)
    assert score["diagnosis_correct"] is True
    assert score["remediation_target_correct"] is False
    assert score["remediation_action_correct"] is False
    assert score["remediation_sop_correct"] is False
    assert score["validation_failure_reason"] == "wrong_target"


def test_wilson_ci_does_not_collapse_for_all_successes():
    rows = [{"system": "full", "diagnosis_correct": True} for _ in range(10)]
    ci = aggregate_ci(rows, ["diagnosis_correct"])
    assert ci["full"]["diagnosis_correct"] == 1.0
    assert ci["full"]["diagnosis_correct_ci95"][0] < 1.0
    assert ci["full"]["diagnosis_correct_ci95"][1] == 1.0


def test_paired_tests_aggregate_repeated_runs_by_scenario():
    rows = []
    for run in range(3):
        rows.extend([
            {"scenario": "a", "run": run, "system": "single", "diagnosis_correct": False},
            {"scenario": "a", "run": run, "system": "full", "diagnosis_correct": True},
            {"scenario": "b", "run": run, "system": "single", "diagnosis_correct": True},
            {"scenario": "b", "run": run, "system": "full", "diagnosis_correct": True},
        ])
    test = paired_mcnemar(rows, "single", "full", "diagnosis_correct")
    effect = paired_bootstrap_effect(rows, "single", "full", "diagnosis_correct")
    assert test["paired_cases"] == 2
    assert test["treatment_only_correct"] == 1
    assert effect["paired_cases"] == 2
    assert effect["mean_difference"] == 0.5


def test_telco_v2_has_balanced_stress_suite():
    ids = list_scenario_ids("telco_v2")
    assert len(ids) == 60
    tags = {tag for sid in ids for tag in get_scenario(sid).stress_tags}
    assert set(STRESS_TAGS) <= tags
    assert len({get_scenario(sid).fault_type for sid in ids}) == 10


def test_telco_v3_scenarios_build_valid_large_topology_incidents():
    ids = list_scenario_ids("telco_v3")
    assert len(ids) == 12
    for sid in ids:
        scenario = get_scenario(sid)
        sim = make_simulator(scenario)
        incident = build_incident(scenario, sim)
        assert scenario.suite == "telco_v3"
        assert len(sim.topology.all()) > 80
        assert incident.alarms
        assert incident.affected_elements
        assert any(k.is_anomalous for k in sim.get_kpis())


def test_scoped_tool_access_blocks_deep_out_of_domain_telemetry():
    scenario = get_scenario("v3_fiber_degradation_a")
    ctx = SessionContext.create(make_simulator(scenario)).scoped({"RAN"})

    diag = json.loads(dispatch(ctx, "run_diagnostic", {"element_id": "FIBER-LINK-01"}))
    assert "outside your domain" in diag["error"]

    logs = json.loads(dispatch(ctx, "query_logs", {"element_id": "FIBER-LINK-01"}))
    assert "outside your domain" in logs["error"]

    kpis = json.loads(dispatch(ctx, "query_kpis", {"element_id": "FIBER-LINK-01"}))
    assert kpis and kpis[0]["note"].startswith("detail restricted")


def test_primary_fault_cleared_is_distinct_from_all_faults_healthy():
    scenario = get_scenario("v3_multifault_fiber_intf")
    sim = make_simulator(scenario)
    assert sim.has_fault_on(scenario.element_id)
    secondary_element = scenario.secondary_faults[0][0]
    assert sim.has_fault_on(secondary_element)

    result = sim.apply_remediation(
        "dispatch field team to restore link and repair fiber FIBER-LINK-03",
        element_id=scenario.element_id,
    )
    assert result["status"] == "resolved"
    assert not sim.has_fault_on(scenario.element_id)
    assert sim.has_fault_on(secondary_element)
    assert not sim.is_healthy()


def test_no_partition_and_no_debate_modes_run_with_stub_llm():
    for mode in ("no_partition", "no_debate"):
        llm = LLMClient(responder=_config_responder, cache_enabled=False)
        result = run("fiber_cut", mode=mode, llm=llm)
        assert result.consensus and result.consensus.faulty_element_id == "FIBER-LINK-01"
        assert result.primary_fault_cleared is True


def _final(payload: dict) -> dict:
    return {"content": json.dumps(payload), "tool_calls": []}


def _tool_call(name: str, args: dict) -> dict:
    return {"content": None, "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}]}


def _config_responder(messages, tools):
    system = messages[0]["content"]
    used_tool = any(m.get("role") == "tool" for m in messages)
    if "triage agent" in system:
        return _final({"severity": "CRITICAL", "suspected_domain": "TRANSPORT",
                       "affected_elements": ["FIBER-LINK-01"], "summary": "transport down"})
    if "knowledge-correlation" in system:
        return _final({"notes": "fiber LOS signature", "relevant_sops": ["SOP-TRANSPORT-FIBER"]})
    if "expert" in system:
        return _final({"faulty_element_id": "FIBER-LINK-01", "fault_type": "FIBER_CUT",
                       "root_cause": "fiber optical link cut", "confidence": 0.9,
                       "rationale": "upstream fiber explains downstream alarms",
                       "evidence": ["optical LOS"]})
    if "remediation agent" in system:
        return _final({"sop_id": "SOP-TRANSPORT-FIBER", "summary": "repair fiber",
                       "steps": ["dispatch", "splice"], "expected_outcome": "link restored",
                       "action": "dispatch field team to repair fiber FIBER-LINK-01",
                       "target_element_id": "FIBER-LINK-01"})
    if "validation agent" in system:
        if not used_tool:
            return _tool_call("apply_remediation", {
                "action": "dispatch field team to repair fiber FIBER-LINK-01",
                "element_id": "FIBER-LINK-01",
            })
        return _final({"resolved": True, "notes": "recovered", "recovered_kpis": ["dl@RAN-CELL-01"]})
    return _final({})

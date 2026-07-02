"""End-to-end wiring of the orchestrator + baseline, driven by a stub LLM.

No network / API key needed: a scripted ``responder`` stands in for the model, so
this verifies the *plumbing* (agent sequencing, tool loop, consensus, validation),
not the model's reasoning quality.
"""
import json

from telco_mas.agents.orchestrator import MultiAgentOrchestrator
from telco_mas.baseline import run_single_agent
from telco_mas.llm import LLMClient
from telco_mas.pipeline import prepare

TARGET = "FIBER-LINK-01"


def _final(payload: dict) -> dict:
    return {"content": json.dumps(payload), "tool_calls": []}


def _tool_call(name: str, args: dict) -> dict:
    return {"content": None, "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}]}


def fake_responder(messages, tools):
    system = messages[0]["content"]
    used_tool = any(m.get("role") == "tool" for m in messages)

    if "triage agent" in system:
        return _final({"severity": "CRITICAL", "suspected_domain": "TRANSPORT",
                       "affected_elements": [TARGET], "summary": "transport branch down"})
    if "knowledge-correlation" in system:
        return _final({"notes": "matches a fiber cut", "relevant_sops": ["SOP-TRANSPORT-FIBER"],
                       "similar_incidents": ["HIST-2024-0142"], "likely_domains": ["TRANSPORT"]})
    if "arbiter" in system:
        return _final({"faulty_element_id": TARGET, "fault_type": "FIBER_CUT",
                       "root_cause": "Fiber cut on FIBER-LINK-01", "confidence": 0.9,
                       "explanation": "transport expert had the strongest evidence"})
    if "remediation agent" in system:
        return _final({"sop_id": "SOP-TRANSPORT-FIBER", "summary": "repair the fiber",
                       "steps": ["dispatch team", "splice fiber"], "expected_outcome": "link back up",
                       "action": "dispatch field team to repair fiber FIBER-LINK-01", "target_element_id": TARGET})
    if "validation agent" in system:
        if not used_tool:
            return _tool_call("apply_remediation", {"action": "repair fiber FIBER-LINK-01", "element_id": TARGET})
        return _final({"resolved": True, "notes": "applied and verified", "recovered_kpis": ["dl_throughput_mbps@RAN-CELL-01"]})
    if "autonomous NOC engineer" in system:  # single-agent baseline
        if not used_tool:
            return _tool_call("apply_remediation", {"action": "repair fiber FIBER-LINK-01", "element_id": TARGET})
        return _final({"root_cause": "Fiber cut on FIBER-LINK-01", "faulty_element_id": TARGET,
                       "fault_type": "FIBER_CUT", "confidence": 0.8, "remediation_summary": "repaired fiber", "resolved": True})
    if "expert" in system:  # a diagnosis expert
        if not used_tool:
            return _tool_call("query_alarms", {})
        conf = 0.9 if "Transport" in system else 0.4
        return _final({"faulty_element_id": TARGET, "fault_type": "FIBER_CUT",
                       "root_cause": "fiber cut upstream", "confidence": conf,
                       "rationale": "LOS explains downstream", "evidence": ["optical -41 dBm"]})
    return _final({})


def test_multi_agent_pipeline_smoke():
    llm = LLMClient(responder=fake_responder, cache_enabled=False)
    ctx, incident, _ = prepare("fiber_cut")
    result = MultiAgentOrchestrator(llm, ctx).run(incident)

    assert result.system == "multi_agent"
    assert len(result.hypotheses) == 3
    assert result.consensus and result.consensus.faulty_element_id == TARGET
    assert result.validation and result.validation.resolved is True
    assert ctx.sim.is_healthy()  # validation actually applied the fix
    assert result.usage.llm_calls > 0 and result.usage.tool_calls > 0
    assert result.trace


def test_single_agent_baseline_smoke():
    llm = LLMClient(responder=fake_responder, cache_enabled=False)
    ctx, incident, _ = prepare("fiber_cut")
    result = run_single_agent(incident, ctx, llm)
    assert result.system == "single_agent"
    assert result.consensus.faulty_element_id == TARGET
    assert result.validation.resolved is True
    assert ctx.sim.is_healthy()

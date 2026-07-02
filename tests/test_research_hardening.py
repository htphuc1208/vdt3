import json

from telco_mas.agents.consensus import tally_votes
from telco_mas.agents.validation import ValidationAgent
from telco_mas.environment.scenarios import get_scenario, make_simulator
from telco_mas.knowledge.retriever import build_default_retriever
from telco_mas.llm import LLMClient
from telco_mas.schemas import Hypothesis, RemediationPlan
from telco_mas.tools.registry import SessionContext


def test_calibrated_consensus_is_deterministic_and_evidence_weighted():
    hypotheses = [
        Hypothesis(
            proposed_by="ran_expert",
            faulty_element_id="RAN-CELL-01",
            fault_type="CELL_OUTAGE",
            confidence=0.9,
            root_cause="cell down",
            evidence=["one alarm"],
        ),
        Hypothesis(
            proposed_by="transport_expert",
            faulty_element_id="FIBER-LINK-01",
            fault_type="FIBER_CUT",
            confidence=0.7,
            root_cause="upstream fiber cut explains downstream alarms",
            rationale="upstream dependency explains blast radius",
            evidence=["diagnostic optical Rx -41 dBm", "SOP-TRANSPORT-FIBER"],
        ),
    ]
    scores1, _ = tally_votes(hypotheses)
    scores2, _ = tally_votes(hypotheses)
    assert scores1 == scores2
    assert scores1["FIBER-LINK-01"] > scores1["RAN-CELL-01"]


def test_validation_does_not_trust_model_claim_without_real_recovery():
    scenario = get_scenario("fiber_cut")
    sim = make_simulator(scenario)
    ctx = SessionContext.create(sim, build_default_retriever())

    def responder(messages, tools):
        return {"content": json.dumps({"resolved": True, "notes": "claimed fixed", "recovered_kpis": []}), "tool_calls": []}

    llm = LLMClient(responder=responder, cache_enabled=False)
    result, _trace, _usage = ValidationAgent(llm, ctx).run(
        RemediationPlan(sop_id="SOP-TRANSPORT-FIBER", summary="claim only"),
        "do nothing",
        "FIBER-LINK-01",
    )
    assert result.resolved is False
    assert not ctx.sim.is_healthy()

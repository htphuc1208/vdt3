import json

from telco_mas.agents.consensus import ConsensusModule, tally_votes
from telco_mas.agents.validation import ValidationAgent
from telco_mas.environment.scenarios import build_incident, get_scenario, make_simulator
from telco_mas.knowledge.fault_ontology import (
    canonicalize_fault_type,
    domain_compatible,
    earliest_specific_root_candidates,
)
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
    assert scores1["RAN-CELL-01"] > scores1["FIBER-LINK-01"]  # no verified signals: confidence fallback

    verified = {"transport_expert": ["FIBER-LINK-01 optical Rx -41 dBm"]}
    coverage = {"transport_expert": 1.0, "ran_expert": 0.1}
    scores3, _ = tally_votes(hypotheses, verified=verified, coverage=coverage)
    scores4, _ = tally_votes(hypotheses, verified=verified, coverage=coverage)
    assert scores3 == scores4
    assert scores3["FIBER-LINK-01"] > scores3["RAN-CELL-01"]


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


def test_fault_ontology_uses_specific_early_root_condition_not_propagated_noise():
    scenario = get_scenario("v3_gps_sync_d1")
    sim = make_simulator(scenario)
    incident = build_incident(scenario, sim)

    candidates = earliest_specific_root_candidates(incident, sim.topology)

    assert [(element, family) for element, family, _ in candidates] == [
        ("RAN-GNB-D1", "GPS_SYNC_LOSS")
    ]
    root = sim.topology.get("RAN-GNB-D1")
    switch = sim.topology.get("AGG-SW-04")
    assert canonicalize_fault_type(
        "INTERFERENCE",
        element=root,
        alarms=incident.alarms,
        root_cause="interference-like symptoms",
    ) == "GPS_SYNC_LOSS"
    assert domain_compatible("GPS_SYNC_LOSS", root) is True
    assert domain_compatible("INTERFERENCE", switch) is False


def test_consensus_verifier_rejects_domain_impossible_high_confidence_candidate():
    scenario = get_scenario("v3_gps_sync_d1")
    sim = make_simulator(scenario)
    incident = build_incident(scenario, sim)
    ctx = SessionContext.create(sim, build_default_retriever())
    hypotheses = [
        Hypothesis(
            proposed_by=expert,
            faulty_element_id="AGG-SW-04",
            fault_type="INTERFERENCE",
            confidence=0.95,
            root_cause="Interference on an aggregation switch",
        )
        for expert in ("ran_expert", "transport_expert", "power_expert", "core_expert")
    ]

    result, trace, usage = ConsensusModule(LLMClient(cache_enabled=False), ctx).run(
        incident,
        hypotheses,
        use_arbiter=False,
    )

    assert result.faulty_element_id == "RAN-GNB-D1"
    assert result.fault_type == "GPS_SYNC_LOSS"
    assert result.root_cause.startswith("GNSS synchronization loss")
    assert trace == []
    assert usage.llm_calls == 0

    ablated, _, _ = ConsensusModule(LLMClient(cache_enabled=False), ctx).run(
        incident,
        hypotheses,
        use_arbiter=False,
        use_evidence_verifier=False,
    )
    assert ablated.faulty_element_id == "AGG-SW-04"

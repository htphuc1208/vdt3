"""Tests for the P0 scientific-hardening changes:
- fair semantic fault-type metric,
- real ablation switches (no_rag / no_consensus),
- KB hold-out + distractors (construct-validity control),
- cache-only offline guard.
"""
import json

import pytest

from telco_mas.agents.orchestrator import MultiAgentOrchestrator, PipelineConfig
from telco_mas.evaluation.metrics import fault_type_match
from telco_mas.knowledge.retriever import build_default_retriever, build_retriever
from telco_mas.llm import LLMClient, LLMError
from telco_mas.pipeline import prepare


# --------------------------------------------------------------------------- #
# 1. Fair semantic fault-type metric (removes the exact-enum artifact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "predicted,true,expected",
    [
        ("FIBER_CUT", "FIBER_CUT", True),               # exact enum
        ("LOS (Loss of optical signal)", "FIBER_CUT", True),   # baseline alarm name
        ("MAINS_FAIL", "POWER_OUTAGE", True),
        ("HIGH_CPU", "CORE_OVERLOAD", True),
        ("CONFIG_MISMATCH", "MISCONFIG", True),
        ("DNS_UNRESOLVED", "DNS_FAILURE", True),
        ("LICENSE_LIMIT", "LICENSE_EXHAUSTION", True),
        ("HIGH_INTERFERENCE", "INTERFERENCE", True),
        ("CARD_FAULT", "HARDWARE_FAILURE", True),
        ("HIGH_LOAD", "CONGESTION", True),
        ("Communication failure", "CELL_OUTAGE", False),  # genuinely vague/wrong family
        ("HARDWARE_FAILURE", "CELL_OUTAGE", False),        # different family
        ("", "FIBER_CUT", False),
    ],
)
def test_fault_type_semantic_match(predicted, true, expected):
    assert fault_type_match(predicted, true) is expected


# --------------------------------------------------------------------------- #
# 2. KB hold-out + distractors
# --------------------------------------------------------------------------- #
def test_holdout_removes_matching_sop():
    r = build_retriever(exclude_sop_ids={"SOP-TRANSPORT-FIBER"},
                        exclude_incident_fault_types={"FIBER_CUT"})
    ids = [d.id for d, _ in r.search("fiber optical loss of signal link down", kind="sop", top_k=5)]
    assert "SOP-TRANSPORT-FIBER" not in ids
    assert all(d.meta.get("fault_type") != "FIBER_CUT"
               for d, _ in r.search("fiber cut", kind="incident", top_k=10))


def test_distractors_enlarge_corpus():
    base = len(build_default_retriever().documents)
    with_distractors = len(build_retriever(include_distractors=True).documents)
    assert with_distractors > base


def test_prepare_holdout_excludes_answer():
    ctx, _incident, _sc = prepare("fiber_cut", holdout=True)
    sop_ids = [d.id for d in ctx.retriever.documents if d.kind == "sop"]
    assert "SOP-TRANSPORT-FIBER" not in sop_ids


# --------------------------------------------------------------------------- #
# 3. Real ablation switches
# --------------------------------------------------------------------------- #
def _final(payload):
    return {"content": json.dumps(payload), "tool_calls": []}


def _ablation_responder(messages, tools):
    system = messages[0]["content"]
    if "triage agent" in system:
        return _final({"severity": "CRITICAL", "suspected_domain": "TRANSPORT",
                       "affected_elements": ["FIBER-LINK-01"], "summary": "down"})
    if "knowledge-correlation" in system:
        return _final({"notes": "fiber-cut-like", "relevant_sops": ["SOP-TRANSPORT-FIBER"]})
    if "remediation agent" in system:
        return _final({"sop_id": "SOP-TRANSPORT-FIBER", "summary": "repair", "steps": ["x"],
                       "action": "dispatch team to repair fiber FIBER-LINK-01", "target_element_id": "FIBER-LINK-01"})
    if "validation agent" in system:
        return _final({"resolved": True, "notes": "ok", "recovered_kpis": []})
    if "expert" in system:
        if "Transport" in system:
            return _final({"faulty_element_id": "FIBER-LINK-01", "fault_type": "FIBER_CUT",
                           "root_cause": "fiber cut", "confidence": 0.9, "evidence": ["-41 dBm"]})
        if "RAN (Radio" in system:
            return _final({"faulty_element_id": "RAN-CELL-01", "fault_type": "CELL_OUTAGE",
                           "root_cause": "cell down", "confidence": 0.5, "evidence": ["alarm"]})
        return _final({"faulty_element_id": "CORE-AMF-01", "fault_type": "MISCONFIG",
                       "root_cause": "core", "confidence": 0.3, "evidence": []})
    return _final({})


def test_no_consensus_takes_most_confident_expert():
    llm = LLMClient(responder=_ablation_responder, cache_enabled=False)
    ctx, incident, _ = prepare("fiber_cut")
    result = MultiAgentOrchestrator(llm, ctx).run(incident, config=PipelineConfig(use_consensus=False))
    assert result.consensus.faulty_element_id == "FIBER-LINK-01"  # the 0.9-confidence expert
    assert result.consensus.vote_breakdown == {}
    assert "ablation" in result.consensus.explanation.lower()


def test_no_rag_skips_correlation():
    llm = LLMClient(responder=_ablation_responder, cache_enabled=False)
    ctx, incident, _ = prepare("fiber_cut")
    result = MultiAgentOrchestrator(llm, ctx).run(incident, config=PipelineConfig(use_rag=False))
    assert result.correlation_notes == ""
    assert not any(step.agent == "correlation" for step in result.trace)


def test_full_pipeline_still_runs_consensus():
    llm = LLMClient(responder=_ablation_responder, cache_enabled=False)
    ctx, incident, _ = prepare("fiber_cut")
    result = MultiAgentOrchestrator(llm, ctx).run(incident)  # default FULL
    assert result.consensus.faulty_element_id == "FIBER-LINK-01"
    assert result.consensus.vote_breakdown  # numeric vote present
    assert result.correlation_notes  # RAG ran


# --------------------------------------------------------------------------- #
# 4. cache-only offline guard
# --------------------------------------------------------------------------- #
def test_cache_only_refuses_live_call_on_miss():
    llm = LLMClient(cache_only=True)
    with pytest.raises(LLMError):
        llm.chat([{"role": "system", "content": "unique-nonce-xyz"},
                  {"role": "user", "content": "no-such-cached-request-123"}])

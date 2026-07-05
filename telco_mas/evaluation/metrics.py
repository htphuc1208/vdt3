"""Scoring a pipeline result against a scenario's ground truth."""
from __future__ import annotations

import json
import re
from statistics import mean

from ..environment.scenarios import Scenario
from ..schemas import PipelineResult


def _norm(text: str | None) -> str:
    return (text or "").strip().upper().replace(" ", "_").replace("-", "_")


def _spaces(text: str | None) -> str:
    return re.sub(r"[_\-]+", " ", (text or "").strip().lower())


# Semantic aliases for each canonical fault family. A prediction is credited if it
# either matches the canonical enum exactly OR mentions any defining alias of the
# TRUE family. This measures diagnostic capability (the correct failure family),
# NOT exact-enum string compliance — so a baseline that says "MAINS_FAIL" or
# "HIGH_CPU" is scored fairly against the multi-agent system that emits the enum.
FAULT_TYPE_ALIASES: dict[str, list[str]] = {
    "FIBER_CUT": ["fiber", "fibre", "los", "loss of signal", "optical", "backhaul cut", "link cut"],
    "CELL_OUTAGE": ["cell down", "cell outage", "cell out of service", "cell unavailable", "rru fail", "rru fault", "radio unit fail"],
    "CONGESTION": ["congestion", "prb", "high load", "radio congestion", "capacity exhaust", "admission control"],
    "MISCONFIG": ["misconfig", "misconfiguration", "config mismatch", "configuration error", "bad config", "config change", "rollback"],
    "HARDWARE_FAILURE": ["hardware", "line card", "linecard", "card fault", "card failure", "crc error", "faulty card"],
    "POWER_OUTAGE": ["power", "mains", "battery", "rectifier", "site power", "ac fail"],
    "CORE_OVERLOAD": ["overload", "cpu", "compute", "resource exhaust", "high cpu", "saturation", "throttl"],
    "DNS_FAILURE": ["dns", "servfail", "resolver", "name resolution", "resolution fail"],
    "LICENSE_EXHAUSTION": ["license", "licence", "entitlement", "license limit", "user limit", "license exhaust"],
    "INTERFERENCE": ["interference", "bler", "noise floor", "uplink noise", "rf interference", "external rf"],
    # telco_v3 masquerade families
    "FIBER_DEGRADATION": ["fiber degrad", "optical degrad", "connector", "attenuat", "rx power", "pre-fec", "optical loss", "fiber"],
    "POWER_BROWNOUT": ["brownout", "voltage sag", "voltage", "power quality", "dc bus", "rectifier"],
    "GPS_SYNC_LOSS": ["gps", "gnss", "sync", "timing", "holdover", "clock drift"],
    "UPF_DEGRADATION": ["upf", "user plane", "packet processing", "core latency", "user-plane"],
}


def fault_type_match(predicted: str | None, true_fault_type: str) -> bool:
    """Fair semantic match of a predicted fault type against the true family."""
    if not predicted:
        return False
    if _norm(predicted) == _norm(true_fault_type):
        return True
    aliases = FAULT_TYPE_ALIASES.get(_norm(true_fault_type), [])
    text = _spaces(predicted)
    return any(alias in text for alias in aliases)


def score_result(result: PipelineResult, scenario: Scenario) -> dict:
    cons = result.consensus
    faulty = cons.faulty_element_id if cons else None
    fault_type = cons.fault_type if cons else None
    root_cause = (cons.root_cause if cons else "") or ""

    # Multi-fault scenarios: any genuinely faulty element counts as correct
    # localization; the fault family is judged against the MATCHED element's fault.
    acceptable = getattr(scenario, "acceptable_elements", (scenario.element_id,))
    secondary = dict(getattr(scenario, "secondary_faults", ()) or ())
    localization = faulty in acceptable
    matched_true_fault = scenario.fault_type
    if faulty and faulty != scenario.element_id and faulty in secondary:
        matched_true_fault = secondary[faulty]
    fault_type_correct = fault_type_match(fault_type, matched_true_fault)
    kws = [k.lower() for k in scenario.root_cause_keywords]
    matched = sum(1 for k in kws if k in root_cause.lower())
    keyword_recall = round(matched / len(kws), 3) if kws else 0.0
    root_cause_correct = matched >= 1
    causal_explanation_correct = matched >= max(1, min(2, len(kws))) and keyword_recall >= 0.3
    # Legacy/loose diagnosis is retained for comparison with earlier runs.
    diagnosis_loose_correct = localization and root_cause_correct
    # Strict diagnosis is the headline metric: element + fault family + causal explanation.
    diagnosis_correct = localization and fault_type_correct and causal_explanation_correct

    remediation_action = result.remediation_action or ""
    remediation_target = result.remediation_target_element_id
    remediation_target_correct = (
        remediation_target == scenario.element_id
        or scenario.element_id.lower() in remediation_action.lower()
    )
    remediation_action_correct = _remediation_action_match(remediation_action, scenario)
    remediation_sop = result.remediation.sop_id if result.remediation else None
    remediation_sop_correct = remediation_sop == scenario.remediation_sop
    # Sim-grounded resolution of the PRIMARY fault (multi-fault safe); fall back
    # to the validation verdict for results predating the field.
    if result.primary_fault_cleared is not None:
        resolved = bool(result.primary_fault_cleared)
    else:
        resolved = bool(result.validation and result.validation.resolved)
    validation_recovery_evidence = resolved and _validation_checked_recovery(result)
    end_to_end_correct = diagnosis_correct and resolved

    return {
        "scenario": scenario.id,
        "suite": scenario.suite,
        "stress_tags": list(scenario.stress_tags),
        "system": result.system,
        "localization": localization,
        "fault_type_correct": fault_type_correct,
        "root_cause_correct": root_cause_correct,
        "causal_explanation_correct": causal_explanation_correct,
        "diagnosis_loose_correct": diagnosis_loose_correct,
        "diagnosis_correct": diagnosis_correct,
        "end_to_end_correct": end_to_end_correct,
        "remediation_target_correct": remediation_target_correct,
        "remediation_action_correct": remediation_action_correct,
        "remediation_sop_correct": remediation_sop_correct,
        "validation_recovery_evidence": validation_recovery_evidence,
        "resolved": resolved,
        "retrieval_hit_at_3": _retrieval_hit_at_k(result, scenario),
        "expert_disagreement": _expert_disagreement(result),
        "arbiter_called": _arbiter_called(result),
        "debate_rounds": result.debate_rounds,
        "debate_called": result.debate_rounds > 0,
        "validation_failure_reason": _validation_failure_reason(
            result,
            target_correct=remediation_target_correct,
            action_correct=remediation_action_correct,
            sop_correct=remediation_sop_correct,
        ),
        "replan_success": result.remediation_attempts > 1 and resolved,
        "remediation_attempts": result.remediation_attempts,
        "keyword_recall": keyword_recall,
        "predicted_element": faulty,
        "true_element": scenario.element_id,
        "predicted_fault_type": fault_type,
        "true_fault_type": scenario.fault_type,
        "predicted_sop": remediation_sop,
        "true_sop": scenario.remediation_sop,
        "remediation_action": remediation_action,
        "remediation_target": remediation_target,
        "total_tokens": result.usage.total_tokens,
        "llm_calls": result.usage.llm_calls,
        "tool_calls": result.usage.tool_calls,
        "latency_s": result.latency_s,
    }


def _remediation_action_match(action: str, scenario: Scenario) -> bool:
    text = (action or "").lower()
    if not text:
        return False
    if scenario.remediation_sop and scenario.remediation_sop.lower() in text:
        return True
    return any(keyword.lower() in text for keyword in scenario.remediation_keywords)


def _retrieval_hit_at_k(result: PipelineResult, scenario: Scenario) -> bool:
    if not scenario.remediation_sop:
        return False
    needle = scenario.remediation_sop.lower()
    for step in result.trace:
        for call in step.tool_calls:
            if call.name == "search_knowledge_base" and needle in call.result_preview.lower():
                return True
    return False


def _expert_disagreement(result: PipelineResult) -> bool:
    elements = {h.faulty_element_id for h in result.hypotheses if h.faulty_element_id}
    return len(elements) > 1


def _arbiter_called(result: PipelineResult) -> bool:
    return any(step.agent == "consensus" for step in result.trace)


def _validation_checked_recovery(result: PipelineResult) -> bool:
    if result.validation and result.validation.recovered_kpis:
        return True
    checked = {"query_alarms", "query_kpis"}
    return any(
        step.agent == "validation" and any(call.name in checked for call in step.tool_calls)
        for step in result.trace
    )


def _validation_failure_reason(
    result: PipelineResult,
    *,
    target_correct: bool,
    action_correct: bool,
    sop_correct: bool,
) -> str:
    if result.validation and result.validation.resolved:
        return ""
    if not target_correct:
        return "wrong_target"
    if not action_correct and not sop_correct:
        return "wrong_action_or_sop"
    if result.validation and not _validation_checked_recovery(result):
        return "missing_validation_evidence"
    status = _apply_remediation_status(result)
    return status or "unresolved_after_correct_attempt"


def _apply_remediation_status(result: PipelineResult) -> str:
    for step in result.trace:
        for call in step.tool_calls:
            if call.name != "apply_remediation":
                continue
            try:
                payload = json.loads(call.result_preview)
            except Exception:
                continue
            status = payload.get("status")
            if status:
                return str(status)
    return ""


def _rate(rows: list[dict], key: str) -> float:
    return round(mean([1.0 if r[key] else 0.0 for r in rows]), 3) if rows else 0.0


def _avg(rows: list[dict], key: str) -> float:
    return round(mean([float(r[key]) for r in rows]), 1) if rows else 0.0


def _sum(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows) if rows else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-scenario scores into per-system summaries."""
    summary: dict[str, dict] = {}
    systems = sorted({r["system"] for r in rows})
    for system in systems:
        srows = [r for r in rows if r["system"] == system]
        summary[system] = {
            "n": len(srows),
            "localization_accuracy": _rate(srows, "localization"),
            "root_cause_accuracy": _rate(srows, "root_cause_correct"),
            "fault_type_accuracy": _rate(srows, "fault_type_correct"),
            "causal_explanation_accuracy": _rate(srows, "causal_explanation_correct"),
            "diagnosis_loose_accuracy": _rate(srows, "diagnosis_loose_correct"),
            "diagnosis_accuracy": _rate(srows, "diagnosis_correct"),
            "end_to_end_accuracy": _rate(srows, "end_to_end_correct"),
            "remediation_target_accuracy": _rate(srows, "remediation_target_correct"),
            "remediation_action_accuracy": _rate(srows, "remediation_action_correct"),
            "remediation_sop_accuracy": _rate(srows, "remediation_sop_correct"),
            "validation_recovery_rate": _rate(srows, "validation_recovery_evidence"),
            "replan_success_rate": _rate(srows, "replan_success"),
            "resolution_rate": _rate(srows, "resolved"),
            "retrieval_hit_at_3_rate": _rate(srows, "retrieval_hit_at_3"),
            "expert_disagreement_rate": _rate(srows, "expert_disagreement"),
            "arbiter_call_rate": _rate(srows, "arbiter_called"),
            "debate_call_rate": _rate(srows, "debate_called"),
            "avg_debate_rounds": _avg(srows, "debate_rounds"),
            "avg_remediation_attempts": _avg(srows, "remediation_attempts"),
            "avg_keyword_recall": _avg(srows, "keyword_recall"),
            "avg_total_tokens": _avg(srows, "total_tokens"),
            "avg_llm_calls": _avg(srows, "llm_calls"),
            "avg_tool_calls": _avg(srows, "tool_calls"),
            "avg_latency_s": _avg(srows, "latency_s"),
            "solved_cases_per_10k_tokens": round(
                (10000.0 * sum(1 for r in srows if r["end_to_end_correct"]) / max(_sum(srows, "total_tokens"), 1.0)),
                3,
            ),
        }
    return summary

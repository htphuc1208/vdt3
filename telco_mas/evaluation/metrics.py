"""Scoring a pipeline result against a scenario's ground truth."""
from __future__ import annotations

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

    localization = faulty == scenario.element_id
    fault_type_correct = fault_type_match(fault_type, scenario.fault_type)
    kws = [k.lower() for k in scenario.root_cause_keywords]
    matched = sum(1 for k in kws if k in root_cause.lower())
    root_cause_correct = matched >= 1
    # "diagnosis correct" = localised the right element AND named the right cause
    diagnosis_correct = localization and root_cause_correct
    resolved = bool(result.validation and result.validation.resolved)

    return {
        "scenario": scenario.id,
        "system": result.system,
        "localization": localization,
        "fault_type_correct": fault_type_correct,
        "root_cause_correct": root_cause_correct,
        "diagnosis_correct": diagnosis_correct,
        "resolved": resolved,
        "keyword_recall": round(matched / len(kws), 3) if kws else 0.0,
        "predicted_element": faulty,
        "true_element": scenario.element_id,
        "predicted_fault_type": fault_type,
        "true_fault_type": scenario.fault_type,
        "total_tokens": result.usage.total_tokens,
        "llm_calls": result.usage.llm_calls,
        "tool_calls": result.usage.tool_calls,
        "latency_s": result.latency_s,
    }


def _rate(rows: list[dict], key: str) -> float:
    return round(mean([1.0 if r[key] else 0.0 for r in rows]), 3) if rows else 0.0


def _avg(rows: list[dict], key: str) -> float:
    return round(mean([float(r[key]) for r in rows]), 1) if rows else 0.0


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
            "diagnosis_accuracy": _rate(srows, "diagnosis_correct"),
            "resolution_rate": _rate(srows, "resolved"),
            "avg_keyword_recall": _avg(srows, "keyword_recall"),
            "avg_total_tokens": _avg(srows, "total_tokens"),
            "avg_llm_calls": _avg(srows, "llm_calls"),
            "avg_tool_calls": _avg(srows, "tool_calls"),
            "avg_latency_s": _avg(srows, "latency_s"),
        }
    return summary

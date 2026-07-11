"""Analyze paired OpenRCA result files across systems."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..evaluation.baseline_selection import resolve_baseline
from ..evaluation.stats import aggregate_ci, paired_bootstrap_effect, paired_mcnemar
from .evaluator import evaluate_prediction
from .formatter import format_prediction
from .schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput
from .task_parser import TASK_FIELDS


REQUIRED_MECHANISM_ABLATIONS = {"no_falsifier", "no_topology", "no_interaction", "no_refinement"}
NO_FIT_WEIGHT_SENTINELS = {"", "(default)", "(default_no_fit)", "default", "default_no_fit", "none", "null"}


def analyze_openrca_results(
    paths: list[str | Path],
    *,
    baseline: str = "strongest_single",
    treatment: str = "shardrca_full",
) -> dict[str, Any]:
    rows = _load_rows(paths)
    all_paired_rows = [
        {
            "case_id": str(row["row_id"]),
            "system": row["system"],
            "strict_correct": bool(row.get("strict_correct")),
            "score": float(row.get("score") or 0.0),
            "total_tokens": int(row.get("total_tokens") or 0),
            "llm_calls": int(row.get("llm_calls") or 0),
            "tool_calls": int(row.get("tool_calls") or 0),
            "latency_s": float(row.get("latency_s") or 0.0),
            "volume_bin": str(row.get("volume_bin") or "unknown"),
        }
        for row in rows
    ]
    contaminated_row_ids = _contaminated_row_ids(paths)
    paired_rows = [
        row for row in all_paired_rows
        if str(row["case_id"]) not in contaminated_row_ids
    ]
    confirmatory_rows = [
        row for row in rows
        if str(row["row_id"]) not in contaminated_row_ids
    ]
    selection_rows = [
        row
        for row in paired_rows
        if row["system"] != "rca_agent_replica"
    ]
    baseline, baseline_selection = resolve_baseline(selection_rows, requested=baseline, treatment=treatment)
    summary = aggregate_ci(paired_rows, ["strict_correct", "score"])
    full_summary = aggregate_ci(all_paired_rows, ["strict_correct", "score"])
    paired = {
        "strict_correct": {
            "mcnemar": paired_mcnemar(paired_rows, baseline, treatment, "strict_correct"),
            "effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "strict_correct"),
        },
        "score": {
            "effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "score"),
        },
    }
    disagreements = _disagreements(confirmatory_rows, baseline=baseline, treatment=treatment)
    comparisons = _role_comparisons(paired_rows, treatment=treatment)
    volume_analysis = _volume_analysis(paired_rows, treatment=treatment)
    candidate_catalog_sources = _candidate_catalog_sources(confirmatory_rows)
    label_derived_candidate_catalog = any(
        bool(source.get("label_derived"))
        or str(source.get("components") or "").lower() == "label_derived"
        for source in candidate_catalog_sources
    )
    overfit_guard = _overfit_guard(
        summary,
        paths,
        treatment=treatment,
        label_derived_candidate_catalog=label_derived_candidate_catalog,
    )
    gate = _protocol_gate(
        summary,
        paired,
        comparisons,
        volume_analysis,
        baseline,
        treatment,
        overfit_guard,
    )
    return {
        "benchmark": "openrca_telecom",
        "paths": [str(path) for path in paths],
        "systems": sorted({row["system"] for row in rows}),
        "baseline": baseline,
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "summary": summary,
        "full_51_summary": full_summary,
        "confirmatory_case_count": len({row["case_id"] for row in paired_rows}),
        "contaminated_row_ids": sorted(contaminated_row_ids),
        "paired": paired,
        "comparisons": comparisons,
        "volume_analysis": volume_analysis,
        "replay_ablations": _replay_ablations(confirmatory_rows),
        "candidate_catalog_sources": candidate_catalog_sources,
        "label_derived_candidate_catalog": label_derived_candidate_catalog,
        "overfit_guard": overfit_guard,
        "clear_win_gate": gate,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:25],
        "usage": _usage(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paired OpenRCA result files")
    parser.add_argument("paths", nargs="+", help="OpenRCA result JSON files, one or more systems")
    parser.add_argument("--baseline", default="strongest_single")
    parser.add_argument("--treatment", default="shardrca_full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    payload = analyze_openrca_results(args.paths, baseline=args.baseline, treatment=args.treatment)
    text = json.dumps(payload, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "clear_win_gate": payload["clear_win_gate"]["passed"],
            "delta": payload["clear_win_gate"]["absolute_delta"],
            "p": payload["clear_win_gate"]["observed_p"],
        }, indent=2))
    else:
        print(text)
    return 0


def _load_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        system = str(data.get("meta", {}).get("system") or Path(path).stem)
        for row in data.get("rows", []):
            item = dict(row)
            item["system"] = str(item.get("system") or system)
            if "row_id" not in item:
                raise ValueError(f"{path} contains a row without row_id")
            rows.append(item)
    return rows


def _contaminated_row_ids(paths: list[str | Path]) -> set[str]:
    contaminated: set[str] = set()
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        prereg_path = payload.get("meta", {}).get("prereg")
        if not prereg_path:
            continue
        prereg_file = Path(str(prereg_path))
        if not prereg_file.exists():
            continue
        prereg = json.loads(prereg_file.read_text(encoding="utf-8"))
        for item in prereg.get("contamination_ledger", []) or []:
            if isinstance(item, dict) and item.get("row_id") is not None:
                contaminated.add(str(item["row_id"]))
    return contaminated


def _disagreements(rows: list[dict[str, Any]], *, baseline: str, treatment: str) -> list[dict[str, Any]]:
    by_row: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_row[str(row["row_id"])][str(row["system"])] = row
    out: list[dict[str, Any]] = []
    for row_id, systems in sorted(by_row.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        if baseline not in systems or treatment not in systems:
            continue
        b = systems[baseline]
        t = systems[treatment]
        if b.get("strict_correct") != t.get("strict_correct") or float(b.get("score") or 0.0) != float(t.get("score") or 0.0):
            out.append({
                "row_id": row_id,
                "task_index": t.get("task_index") or b.get("task_index"),
                "baseline_score": b.get("score"),
                "treatment_score": t.get("score"),
                "baseline_strict": b.get("strict_correct"),
                "treatment_strict": t.get("strict_correct"),
                "baseline_prediction": b.get("prediction"),
                "treatment_prediction": t.get("prediction"),
            })
    return out


def _usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for system in sorted({str(row["system"]) for row in rows}):
        sr = [row for row in rows if str(row["system"]) == system]
        out[system] = {
            "n": len(sr),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in sr),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in sr),
            "tool_calls": sum(int(row.get("tool_calls") or 0) for row in sr),
            "avg_latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in sr) / len(sr), 3) if sr else 0.0,
        }
    return out


def _candidate_catalog_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
        source = artifacts.get("candidate_catalog_source")
        if not isinstance(source, dict):
            continue
        key = json.dumps(source, sort_keys=True, default=str)
        if key in seen:
            continue
        out.append(dict(source))
        seen.add(key)
    return out


def _overfit_guard(
    summary: dict[str, dict[str, Any]],
    paths: list[str | Path],
    *,
    treatment: str,
    label_derived_candidate_catalog: bool,
) -> dict[str, Any]:
    systems = set(summary)
    missing_ablations = sorted(REQUIRED_MECHANISM_ABLATIONS - systems)
    weight_sources = _weight_sources(paths)
    weights_declared = bool(weight_sources) and all(item["declared"] for item in weight_sources)
    weights_no_fit = weights_declared and all(_is_no_fit_weight_source(item["source"]) for item in weight_sources)
    checks = {
        "treatment_present": treatment in systems,
        "required_mechanism_ablations": not missing_ablations,
        "no_interaction_ablation": "no_interaction" in systems,
        "weights_declared_no_fit": weights_no_fit,
        "candidate_catalog_not_label_derived": not label_derived_candidate_catalog,
    }
    reasons = []
    if not checks["treatment_present"]:
        reasons.append(f"treatment system is missing from paired rows: {treatment}")
    if missing_ablations:
        reasons.append(f"missing required mechanism ablations: {', '.join(missing_ablations)}")
    if not weights_no_fit:
        reasons.append("result metadata must declare default no-fit fusion weights; fitted artifacts are not claim evidence")
    if label_derived_candidate_catalog:
        reasons.append("candidate catalog is label-derived")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "required_ablations": sorted(REQUIRED_MECHANISM_ABLATIONS),
        "missing_ablations": missing_ablations,
        "fusion_weight_sources": weight_sources,
        "reasons": reasons,
        "required": (
            "claim evidence must use default no-fit weights, include no_interaction/no_topology/"
            "no_falsifier/no_refinement ablations, and avoid label-derived candidate catalogs"
        ),
    }


def _weight_sources(paths: list[str | Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        p = Path(path)
        declared = False
        source = "(missing)"
        try:
            meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
        except Exception:
            meta = {}
        if isinstance(meta, dict) and "shardrca_weights" in meta:
            declared = True
            source = str(meta.get("shardrca_weights") or "")
        out.append({"path": str(p), "declared": declared, "source": source})
    return out


def _is_no_fit_weight_source(source: str) -> bool:
    return str(source or "").strip().lower() in NO_FIT_WEIGHT_SENTINELS


def _clear_win_gate(
    summary: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    baseline_acc = float(summary.get(baseline, {}).get("strict_correct") or 0.0)
    treatment_acc = float(summary.get(treatment, {}).get("strict_correct") or 0.0)
    absolute_delta = treatment_acc - baseline_acc
    error_reduction = (
        absolute_delta / (1.0 - baseline_acc)
        if baseline_acc < 1.0
        else 0.0
    )
    p_value = paired["strict_correct"]["mcnemar"]["p_value_exact"]
    effect_ok = absolute_delta >= 0.10 or error_reduction >= 0.20
    return {
        "passed": bool(effect_ok and p_value <= 0.05),
        "required": (
            "treatment beats baseline by >=0.10 absolute strict accuracy or >=20% relative "
            "error reduction, with exact paired p <= 0.05"
        ),
        "baseline": baseline,
        "treatment": treatment,
        "baseline_strict_accuracy": round(baseline_acc, 4),
        "treatment_strict_accuracy": round(treatment_acc, 4),
        "absolute_delta": round(absolute_delta, 4),
        "relative_error_reduction": round(error_reduction, 4),
        "observed_p": p_value,
    }


def _role_comparisons(rows: list[dict[str, Any]], *, treatment: str) -> dict[str, Any]:
    roles = {
        "operational_single": "single_react_sc",
        "architecture_baseline": "rca_agent_replica",
    }
    available = {str(row["system"]) for row in rows}
    comparisons: dict[str, Any] = {}
    for role, baseline in roles.items():
        if baseline not in available or treatment not in available:
            continue
        comparisons[role] = {
            "baseline": baseline,
            "strict": {
                "mcnemar": paired_mcnemar(rows, baseline, treatment, "strict_correct"),
                "effect": paired_bootstrap_effect(rows, baseline, treatment, "strict_correct"),
            },
            "score": {
                "effect": paired_bootstrap_effect(rows, baseline, treatment, "score"),
            },
        }
    confirmatory = [
        item
        for role, item in comparisons.items()
        if role in {"operational_single", "architecture_baseline"}
    ]
    adjusted = _holm_adjust([
        float(item["strict"]["mcnemar"]["p_value_exact"])
        for item in confirmatory
    ])
    for item, adjusted_p in zip(confirmatory, adjusted):
        item["strict"]["holm_adjusted_p"] = adjusted_p
    return comparisons


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        current = min(1.0, (total - rank) * p_values[index])
        running = max(running, current)
        adjusted[index] = running
    return adjusted


def _protocol_gate(
    summary: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    comparisons: dict[str, Any],
    volume_analysis: dict[str, Any],
    baseline: str,
    treatment: str,
    overfit_guard: dict[str, Any],
) -> dict[str, Any]:
    legacy = _clear_win_gate(summary, paired, baseline, treatment)
    required_roles = {"operational_single", "architecture_baseline"}
    missing_roles = sorted(required_roles - set(comparisons))
    if missing_roles or not overfit_guard.get("passed"):
        legacy.update({
            "passed": False,
            "claim_level": "legacy_or_incomplete",
            "strong_mechanism_passed": False,
            "protocol_complete": False,
            "missing_protocol_roles": missing_roles,
            "overfit_guard_passed": bool(overfit_guard.get("passed")),
            "overfit_guard_reasons": list(overfit_guard.get("reasons") or []),
        })
        return legacy

    confirmatory_checks = {}
    for role in sorted(required_roles):
        item = comparisons[role]
        effect = float(item["strict"]["effect"]["mean_difference"])
        baseline_accuracy = _system_accuracy(summary, item["baseline"])
        error_reduction = effect / (1.0 - baseline_accuracy) if baseline_accuracy < 1.0 else 0.0
        adjusted_p = float(item["strict"].get("holm_adjusted_p", 1.0))
        confirmatory_checks[role] = {
            "baseline": item["baseline"],
            "absolute_delta": round(effect, 4),
            "relative_error_reduction": round(error_reduction, 4),
            "holm_adjusted_p": adjusted_p,
            "passed": bool((effect >= 0.10 or error_reduction >= 0.20) and adjusted_p <= 0.05),
        }
    operational_passed = all(item["passed"] for item in confirmatory_checks.values())
    high_volume_positive = bool(volume_analysis.get("positive_against_confirmatory"))
    strong = bool(operational_passed and high_volume_positive)
    architecture = confirmatory_checks["architecture_baseline"]
    return {
        "passed": bool(operational_passed and overfit_guard.get("passed")),
        "protocol_complete": True,
        "overfit_guard_passed": bool(overfit_guard.get("passed")),
        "claim_level": "strong_mechanism" if strong else ("operational_budget" if operational_passed else "none"),
        "strong_mechanism_passed": strong,
        "confirmatory": confirmatory_checks,
        "high_volume_positive": high_volume_positive,
        "required": (
            "MAS must pass Holm-adjusted effect/significance gates against both the operational "
            "single and RCA-Agent; strong mechanism additionally requires a positive high-volume-bin delta"
        ),
        "baseline": architecture["baseline"],
        "treatment": treatment,
        "absolute_delta": architecture["absolute_delta"],
        "relative_error_reduction": architecture["relative_error_reduction"],
        "observed_p": architecture["holm_adjusted_p"],
    }


def _system_accuracy(summary: dict[str, dict[str, Any]], system: str) -> float:
    return float(summary.get(system, {}).get("strict_correct") or 0.0)


def _volume_analysis(rows: list[dict[str, Any]], *, treatment: str) -> dict[str, Any]:
    high = [row for row in rows if row.get("volume_bin") == "high"]
    systems = sorted({str(row["system"]) for row in high})
    accuracy = {}
    for system in systems:
        values = [
            1.0 if row.get("strict_correct") else 0.0
            for row in high
            if row["system"] == system
        ]
        accuracy[system] = sum(values) / len(values) if values else 0.0
    treatment_accuracy = accuracy.get(treatment)
    deltas = {
        baseline: round(treatment_accuracy - score, 4)
        for baseline, score in accuracy.items()
        if baseline != treatment and treatment_accuracy is not None
    }
    confirmatory = [
        delta
        for baseline, delta in deltas.items()
        if baseline in {"single_react_sc", "rca_agent_replica"}
    ]
    return {
        "bin": "high",
        "case_count": len({str(row["case_id"]) for row in high}),
        "accuracy": accuracy,
        "treatment_deltas": deltas,
        "positive_against_confirmatory": bool(confirmatory and all(delta > 0 for delta in confirmatory)),
    }


def _replay_ablations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    replay_rows: dict[str, list[float]] = {
        "no_falsifier": [],
        "additive_fusion": [],
    }
    for row in rows:
        if str(row.get("system")) != "shardrca_full":
            continue
        artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
        scoring_points = str(row.get("scoring_points") or "")
        task_index = str(row.get("task_index") or "")
        pre = artifacts.get("pre_falsifier_winner")
        if isinstance(pre, dict):
            prediction = _prediction_from_candidate(pre, task_index)
            _, _, score = evaluate_prediction(format_prediction(prediction), scoring_points)
            replay_rows["no_falsifier"].append(score)
        additive = _additive_candidate(artifacts)
        if additive is not None:
            prediction = _prediction_from_candidate(additive, task_index)
            _, _, score = evaluate_prediction(format_prediction(prediction), scoring_points)
            replay_rows["additive_fusion"].append(score)
    return {
        name: {
            "n": len(scores),
            "strict_accuracy": (
                round(sum(score == 1.0 for score in scores) / len(scores), 4)
                if scores
                else 0.0
            ),
            "partial_accuracy": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "extra_llm_calls": 0,
        }
        for name, scores in replay_rows.items()
    }


def _additive_candidate(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    totals: dict[tuple[str, str], float] = {}
    for distribution in artifacts.get("worker_distributions", []) or []:
        if not isinstance(distribution, dict):
            continue
        for candidate in distribution.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            pair = (str(candidate.get("component") or ""), str(candidate.get("reason_family") or ""))
            if not all(pair):
                continue
            totals[pair] = totals.get(pair, 0.0) + float(candidate.get("probability") or 0.0)
    if not totals:
        return None
    component, reason = max(totals, key=lambda pair: (totals[pair], pair))
    fused = next(
        (
            candidate
            for candidate in artifacts.get("fusion_candidates", []) or []
            if candidate.get("component") == component and candidate.get("reason") == reason
        ),
        {},
    )
    return {
        "component": component,
        "reason": reason,
        "occurrence_time": fused.get("occurrence_time"),
    }


def _prediction_from_candidate(candidate: dict[str, Any], task_index: str) -> OpenRCAPredictionOutput:
    requested = set(TASK_FIELDS.get(task_index, ()))
    return OpenRCAPredictionOutput(root_causes=[OpenRCAPredictionItem(
        root_cause_occurrence_datetime=(
            candidate.get("occurrence_time")
            if "root cause occurrence datetime" in requested
            else None
        ),
        root_cause_component=(
            candidate.get("component")
            if "root cause component" in requested
            else None
        ),
        root_cause_reason=(
            candidate.get("reason")
            if "root cause reason" in requested
            else None
        ),
    )])


if __name__ == "__main__":
    raise SystemExit(main())

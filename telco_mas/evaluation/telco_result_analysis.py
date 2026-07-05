"""Paired analysis for synthetic telco benchmark result JSONs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .stats import paired_bootstrap_effect, paired_mcnemar


PRIMARY_METRIC = "diagnosis_correct"
SECONDARY_METRICS = [
    "end_to_end_correct",
    "localization",
    "fault_type_correct",
    "causal_explanation_correct",
    "resolved",
]


def analyze_telco_results(
    path: str | Path,
    *,
    baseline: str = "single",
    treatment: str = "full",
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    paired = _paired_rows(rows, baseline, treatment)
    metrics = [PRIMARY_METRIC, *SECONDARY_METRICS]
    paired_metrics = {
        metric: {
            "mcnemar": paired_mcnemar(rows, baseline, treatment, metric),
            "effect": paired_bootstrap_effect(rows, baseline, treatment, metric),
        }
        for metric in metrics
    }
    primary_effect = paired_metrics[PRIMARY_METRIC]["effect"]["mean_difference"]
    primary_p = paired_metrics[PRIMARY_METRIC]["mcnemar"]["p_value_exact"]
    usage = _usage(rows, baseline, treatment)
    integrity = _confirmatory_integrity(payload)
    statistical_gate = primary_effect >= 0.10 and primary_p <= 0.05
    return {
        "benchmark": "synthetic_telco",
        "source": str(path),
        "suite": payload.get("meta", {}).get("suite"),
        "systems": payload.get("meta", {}).get("systems"),
        "baseline": baseline,
        "treatment": treatment,
        "primary_metric": PRIMARY_METRIC,
        "summary": payload.get("summary", {}),
        "paired_metrics": paired_metrics,
        "clear_win_gate": {
            "passed": statistical_gate and integrity["passed"],
            "statistical_gate_passed": statistical_gate,
            "integrity_gate_passed": integrity["passed"],
            "required": (
                "treatment-baseline strict diagnosis delta >= 0.10, exact paired p <= 0.05, "
                "and frozen-protocol integrity for telco_v4"
            ),
            "observed_delta": primary_effect,
            "observed_p": primary_p,
        },
        "confirmatory_integrity": integrity,
        "stratified_diagnostics": {
            "fault_family": _stratified(rows, baseline, treatment, lambda row: row.get("true_fault_type") or "UNKNOWN"),
            "nuisance_profile": _stratified(rows, baseline, treatment, _nuisance_profile),
        },
        "usage": usage,
        "disagreements": _disagreements(paired, baseline, treatment),
        "limitations": [
            "Simulator/synthetic result; not headline real-telco evidence.",
            "Do not use if benchmark readiness lacks matching preregistration.",
            "If clear_win_gate is false, report as negative or exploratory evidence.",
            "Fault-family and nuisance-profile strata are descriptive diagnostics, not separately tested claims.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paired synthetic telco benchmark results.")
    parser.add_argument("path")
    parser.add_argument("--baseline", default="single")
    parser.add_argument("--treatment", default="full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    analysis = analyze_telco_results(args.path, baseline=args.baseline, treatment=args.treatment)
    text = json.dumps(analysis, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "clear_win_gate": analysis["clear_win_gate"]["passed"],
            "delta": analysis["clear_win_gate"]["observed_delta"],
            "p": analysis["clear_win_gate"]["observed_p"],
        }, indent=2))
    else:
        print(text)
    return 0


def _paired_rows(rows: list[dict[str, Any]], baseline: str, treatment: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        system = row.get("system")
        if system not in {baseline, treatment}:
            continue
        scenario = str(row.get("scenario") or row.get("case_id") or row.get("id"))
        run = int(row.get("run") or 0)
        pair_id = f"{scenario}#run={run}"
        out.setdefault(pair_id, {})[str(system)] = row
    return {scenario: systems for scenario, systems in out.items() if baseline in systems and treatment in systems}


def _disagreements(
    paired: dict[str, dict[str, dict[str, Any]]],
    baseline: str,
    treatment: str,
) -> list[dict[str, Any]]:
    out = []
    for scenario, systems in sorted(paired.items()):
        b = systems[baseline]
        t = systems[treatment]
        if bool(b.get(PRIMARY_METRIC)) == bool(t.get(PRIMARY_METRIC)):
            continue
        out.append({
            "scenario": scenario,
            "winner": treatment if t.get(PRIMARY_METRIC) else baseline,
            "true_element": t.get("true_element") or b.get("true_element"),
            "baseline": _row_brief(b),
            "treatment": _row_brief(t),
        })
    return out


def _row_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnosis_correct": row.get("diagnosis_correct"),
        "end_to_end_correct": row.get("end_to_end_correct"),
        "predicted_element": row.get("predicted_element"),
        "predicted_fault_type": row.get("predicted_fault_type"),
        "keyword_recall": row.get("keyword_recall"),
        "resolved": row.get("resolved"),
        "tokens": row.get("total_tokens"),
    }


def _usage(rows: list[dict[str, Any]], baseline: str, treatment: str) -> dict[str, Any]:
    def avg(system: str, key: str) -> float:
        values = [float(row.get(key) or 0.0) for row in rows if row.get("system") == system]
        return round(sum(values) / len(values), 3) if values else 0.0

    b_tokens = avg(baseline, "total_tokens")
    t_tokens = avg(treatment, "total_tokens")
    return {
        baseline: {
            "avg_total_tokens": b_tokens,
            "avg_llm_calls": avg(baseline, "llm_calls"),
            "avg_tool_calls": avg(baseline, "tool_calls"),
            "avg_latency_s": avg(baseline, "latency_s"),
        },
        treatment: {
            "avg_total_tokens": t_tokens,
            "avg_llm_calls": avg(treatment, "llm_calls"),
            "avg_tool_calls": avg(treatment, "tool_calls"),
            "avg_latency_s": avg(treatment, "latency_s"),
        },
        "treatment_to_baseline_token_ratio": round(t_tokens / b_tokens, 3) if b_tokens else 0.0,
    }


def _stratified(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    labeler,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(labeler(row))].append(row)
    output: dict[str, Any] = {}
    for label, stratum_rows in sorted(grouped.items()):
        by_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in stratum_rows:
            system = str(row.get("system"))
            if system not in {baseline, treatment}:
                continue
            case_id = str(row.get("scenario") or row.get("case_id") or row.get("id"))
            by_case[case_id][system].append(1.0 if row.get(PRIMARY_METRIC) else 0.0)
        paired = [
            (mean(values[baseline]), mean(values[treatment]))
            for values in by_case.values()
            if baseline in values and treatment in values
        ]
        if not paired:
            continue
        baseline_rate = mean(item[0] for item in paired)
        treatment_rate = mean(item[1] for item in paired)
        output[label] = {
            "paired_scenarios": len(paired),
            f"{baseline}_accuracy": round(baseline_rate, 4),
            f"{treatment}_accuracy": round(treatment_rate, 4),
            "treatment_minus_baseline": round(treatment_rate - baseline_rate, 4),
        }
    return output


def _nuisance_profile(row: dict[str, Any]) -> str:
    tags = set(row.get("stress_tags") or [])
    if "distractor_alarms" in tags:
        return "alarm_distractor"
    if "missing_noisy_telemetry" in tags:
        return "incomplete_alarm_view"
    if "no_exact_sop" in tags:
        return "kb_holdout"
    return "complete"


def _confirmatory_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    meta = payload.get("meta", {})
    suites = meta.get("suite") or []
    if isinstance(suites, str):
        suites = [suites]
    if "telco_v4" not in suites:
        return {
            "applicable": False,
            "passed": True,
            "checks": {"telco_v4_protocol": "not applicable"},
        }

    checks: dict[str, bool] = {}
    dataset_meta = meta.get("frozen_dataset") or {}
    prereg_path = meta.get("preregistration")
    checks["dataset_sha_recorded"] = bool(dataset_meta.get("sha256"))
    checks["content_sha_recorded"] = bool(dataset_meta.get("content_sha256"))
    checks["design_recorded"] = bool(dataset_meta.get("design"))
    checks["algorithm_id_recorded"] = bool(meta.get("algorithm_id"))
    checks["cache_disabled"] = meta.get("cache") is False
    checks["preregistration_recorded"] = bool(prereg_path)

    if prereg_path:
        try:
            prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
            checks["prereg_status_frozen"] = prereg.get("status") == "frozen"
            checks["dataset_sha_matches_prereg"] = (
                prereg.get("dataset", {}).get("sha256") == dataset_meta.get("sha256")
            )
            checks["algorithm_matches_prereg"] = (
                prereg.get("algorithm", {}).get("id") == meta.get("algorithm_id")
            )
            checks["systems_match_prereg"] = prereg.get("systems") == meta.get("systems")
            checks["runs_match_prereg"] = (
                prereg.get("execution", {}).get("runs") == meta.get("runs")
            )
            checks["model_matches_prereg"] = (
                prereg.get("model", {}).get("name") == meta.get("model")
            )
            checks["temperature_matches_prereg"] = (
                prereg.get("model", {}).get("temperature") == meta.get("temperature")
            )
            checks["base_url_matches_prereg"] = (
                str(prereg.get("model", {}).get("base_url") or "").rstrip("/")
                == str(meta.get("base_url") or "").rstrip("/")
            )
            checks["max_tool_iters_match_prereg"] = (
                prereg.get("model", {}).get("max_tool_iters") == meta.get("max_tool_iters")
            )
            checks["scenario_selection_matches_prereg"] = (
                prereg.get("row_selection", {}).get("source_scenario_ids")
                == meta.get("scenarios")
            )
            expected_pairs = {
                (str(scenario), str(system), run)
                for scenario in (meta.get("scenarios") or [])
                for system in (meta.get("systems") or [])
                for run in range(int(meta.get("runs") or 0))
            }
            observed_pairs = {
                (
                    str(row.get("scenario") or row.get("case_id") or row.get("id")),
                    str(row.get("system")),
                    int(row.get("run") or 0),
                )
                for row in payload.get("rows", [])
            }
            checks["result_rows_complete_and_unique"] = (
                observed_pairs == expected_pairs
                and len(payload.get("rows", [])) == len(expected_pairs)
            )
        except (OSError, json.JSONDecodeError):
            checks["preregistration_readable"] = False

    dataset_path = dataset_meta.get("path")
    if dataset_path:
        try:
            checks["dataset_file_matches_sha"] = (
                _sha256_file(Path(dataset_path)) == dataset_meta.get("sha256")
            )
        except OSError:
            checks["dataset_file_readable"] = False

    return {
        "applicable": True,
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

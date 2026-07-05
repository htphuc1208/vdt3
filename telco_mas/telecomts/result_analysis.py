"""Event-level paired analysis for TelecomTS RCA results."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from ..evaluation.baseline_selection import STRONGEST_SINGLE_ALIASES, is_single_system
from ..evaluation.stats import paired_bootstrap_effect, paired_mcnemar, wilson_ci
from .dataset import RCA_CLASSES


def analyze_results(
    paths: list[str | Path],
    *,
    baseline: str = "strongest_single",
    treatment: str = "telecomts_shardrca_full",
) -> dict[str, Any]:
    payloads = [_load_payload(path) for path in paths]
    rows = [dict(row) for payload in payloads for row in payload.get("rows", [])]
    if not rows:
        raise ValueError("No TelecomTS result rows found")
    baseline, baseline_selection = _resolve_baseline(rows, requested=baseline, treatment=treatment)
    paired_rows = _paired_rows(rows, baseline, treatment)
    if not paired_rows:
        raise ValueError(f"No paired TelecomTS rows for {baseline} and {treatment}")
    summary = _summary(rows)
    paired = {
        "strict_correct": {
            "mcnemar": paired_mcnemar(paired_rows, baseline, treatment, "strict_correct"),
            "micro_effect": paired_bootstrap_effect(
                paired_rows, baseline, treatment, "strict_correct"
            ),
            "macro_effect": _paired_macro_effect(paired_rows, baseline, treatment),
        },
    }
    modes = sorted({str(payload.get("meta", {}).get("mode") or "unknown") for payload in payloads})
    splits = sorted({str(payload.get("meta", {}).get("split") or "unknown") for payload in payloads})
    event_unit = all(payload.get("meta", {}).get("event_unit") is True for payload in payloads)
    frozen_prereg = _all_frozen_prereg(payloads)
    llm_mode = modes == ["llm"]
    test_split = splits == ["test"]
    disagreements = _disagreements(rows, baseline, treatment)
    return {
        "benchmark": "telecomts",
        "paths": [str(path) for path in paths],
        "systems": sorted({str(row["system"]) for row in rows}),
        "baseline": baseline,
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "modes": modes,
        "splits": splits,
        "llm_mode": llm_mode,
        "test_split": test_split,
        "event_unit": event_unit,
        "frozen_prereg": frozen_prereg,
        "evidence_mode": (
            "confirmatory_synthetic_test"
            if llm_mode and test_split and event_unit and frozen_prereg
            else "development_or_nonclaim"
        ),
        "summary": summary,
        "paired": paired,
        "clear_win_gate": _clear_win_gate(summary, paired, baseline, treatment),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:50],
        "usage": _usage(rows),
        "limitations": [
            "TelecomTS RCA uses ten synthetic anomaly transformations over measured 5G testbed KPIs.",
            "The upstream real jamming sessions are excluded from this root-cause classification task.",
            "The source-session-held-out test contains 39 independent anomaly events; two classes have one event each.",
            "Macro accuracy is therefore primary, but per-class uncertainty remains large.",
            "single_equal_calls matches MAS call count, not exact tokens; measured token totals must be reported.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paired TelecomTS event-level results.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--baseline", default="strongest_single")
    parser.add_argument("--treatment", default="telecomts_shardrca_full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        payload = analyze_results(args.paths, baseline=args.baseline, treatment=args.treatment)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    text = json.dumps(payload, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "baseline": payload["baseline"],
            "clear_win_gate": payload["clear_win_gate"]["passed"],
            "macro_delta": payload["clear_win_gate"]["absolute_delta"],
            "p": payload["clear_win_gate"]["observed_p"],
            "evidence_mode": payload["evidence_mode"],
        }, indent=2))
    else:
        print(text)
    return 0


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("suite") != "telecomts":
        raise ValueError(f"Not a TelecomTS result file: {path}")
    return payload


def _resolve_baseline(
    rows: list[dict[str, Any]],
    *,
    requested: str,
    treatment: str,
) -> tuple[str, dict[str, Any]]:
    if requested not in STRONGEST_SINGLE_ALIASES:
        return requested, {"requested": requested, "resolved": requested, "method": "explicit"}
    candidates = []
    treatment_cases = {str(row["case_id"]) for row in rows if row.get("system") == treatment}
    for system in sorted({str(row["system"]) for row in rows}):
        if system == treatment or not is_single_system(system):
            continue
        system_rows = [
            row for row in rows
            if row.get("system") == system and str(row.get("case_id")) in treatment_cases
        ]
        metrics = _system_metrics(system_rows)
        if not system_rows:
            continue
        candidates.append({
            "system": system,
            "macro_accuracy": metrics["macro_accuracy"],
            "micro_accuracy": metrics["micro_accuracy"],
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in system_rows),
            "paired_cases": len({str(row["case_id"]) for row in system_rows}),
        })
    if not candidates:
        raise ValueError(f"No single-agent baseline is paired with treatment '{treatment}'")
    selected = sorted(candidates, key=lambda item: (
        -float(item["macro_accuracy"]),
        -float(item["micro_accuracy"]),
        int(item["total_tokens"]),
        str(item["system"]),
    ))[0]
    return str(selected["system"]), {
        "requested": requested,
        "resolved": selected["system"],
        "method": "strongest_single_by_macro_then_micro_then_tokens",
        "candidates": candidates,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        system: _system_metrics([row for row in rows if str(row["system"]) == system])
        for system in sorted({str(row["system"]) for row in rows})
    }


def _system_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_class = {}
    for name in RCA_CLASSES:
        class_rows = [row for row in rows if row.get("target") == name]
        correct = sum(1 for row in class_rows if row.get("strict_correct"))
        center, lo, hi = wilson_ci(correct, len(class_rows))
        per_class[name] = {
            "n": len(class_rows),
            "accuracy": center,
            "accuracy_ci95": [lo, hi],
        }
    observed = [item["accuracy"] for item in per_class.values() if item["n"]]
    correct = sum(1 for row in rows if row.get("strict_correct"))
    micro, micro_lo, micro_hi = wilson_ci(correct, len(rows))
    macro, macro_lo, macro_hi = _macro_jeffreys_interval(rows)
    return {
        "n": len(rows),
        "micro_accuracy": micro,
        "micro_accuracy_ci95": [micro_lo, micro_hi],
        "macro_accuracy": macro if observed else 0.0,
        "macro_accuracy_ci95": [macro_lo, macro_hi],
        "macro_accuracy_interval_method": "Jeffreys beta-binomial posterior by class",
        "per_class": per_class,
        "errors": sum(1 for row in rows if row.get("error")),
    }


def _macro_jeffreys_interval(
    rows: list[dict[str, Any]],
    *,
    samples: int = 2000,
    seed: int = 20260704,
) -> tuple[float, float, float]:
    by_class = {
        name: [bool(row.get("strict_correct")) for row in rows if row.get("target") == name]
        for name in RCA_CLASSES
    }
    available = [values for values in by_class.values() if values]
    if not available:
        return 0.0, 0.0, 0.0
    center = mean(sum(values) / len(values) for values in available)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(
            rng.betavariate(0.5 + sum(values), 0.5 + len(values) - sum(values))
            for values in available
        ))
    estimates.sort()
    return (
        round(center, 4),
        round(estimates[int(0.025 * samples)], 4),
        round(estimates[min(samples - 1, int(0.975 * samples))], 4),
    )


def _paired_macro_effect(
    rows: list[dict[str, Any]],
    baseline: str,
    treatment: str,
    *,
    samples: int = 2000,
    seed: int = 20260704,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["system"])] = row
    by_class: dict[str, list[float]] = defaultdict(list)
    for systems in by_case.values():
        if baseline not in systems or treatment not in systems:
            continue
        b = systems[baseline]
        t = systems[treatment]
        target = str(t.get("target") or b.get("target"))
        by_class[target].append(
            (1.0 if t.get("strict_correct") else 0.0)
            - (1.0 if b.get("strict_correct") else 0.0)
        )
    available = [values for name, values in by_class.items() if name in RCA_CLASSES and values]
    if not available:
        return {"mean_difference": 0.0, "mean_difference_ci95": [0.0, 0.0], "paired_events": 0}
    center = mean(mean(values) for values in available)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(
            mean(values[rng.randrange(len(values))] for _ in values)
            for values in available
        ))
    estimates.sort()
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": "macro_root_cause_accuracy",
        "paired_events": sum(len(values) for values in available),
        "class_count": len(available),
        "mean_difference": round(center, 4),
        "mean_difference_ci95": [
            round(estimates[int(0.025 * samples)], 4),
            round(estimates[min(samples - 1, int(0.975 * samples))], 4),
        ],
    }


def _paired_rows(rows: list[dict[str, Any]], baseline: str, treatment: str) -> list[dict[str, Any]]:
    by_case: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_case[str(row["case_id"])].add(str(row["system"]))
    paired_cases = {
        case_id for case_id, systems in by_case.items()
        if baseline in systems and treatment in systems
    }
    return [
        row for row in rows
        if str(row["case_id"]) in paired_cases and str(row["system"]) in {baseline, treatment}
    ]


def _clear_win_gate(
    summary: dict[str, Any],
    paired: dict[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    baseline_macro = float(summary.get(baseline, {}).get("macro_accuracy") or 0.0)
    treatment_macro = float(summary.get(treatment, {}).get("macro_accuracy") or 0.0)
    delta = treatment_macro - baseline_macro
    p_value = paired["strict_correct"]["mcnemar"]["p_value_exact"]
    return {
        "passed": bool(delta >= 0.10 and p_value <= 0.05),
        "required": "event-level macro accuracy delta >=0.10 and exact paired McNemar p <=0.05",
        "baseline": baseline,
        "treatment": treatment,
        "baseline_macro_accuracy": round(baseline_macro, 4),
        "treatment_macro_accuracy": round(treatment_macro, 4),
        "absolute_delta": round(delta, 4),
        "observed_p": p_value,
        "observed_delta": round(delta, 4),
    }


def _disagreements(
    rows: list[dict[str, Any]], baseline: str, treatment: str
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["system"])] = row
    out = []
    for case_id, systems in sorted(by_case.items()):
        if baseline not in systems or treatment not in systems:
            continue
        b, t = systems[baseline], systems[treatment]
        if b.get("strict_correct") != t.get("strict_correct") or b.get("predicted") != t.get("predicted"):
            out.append({
                "case_id": case_id,
                "target": t.get("target"),
                "baseline_prediction": b.get("predicted"),
                "treatment_prediction": t.get("predicted"),
                "baseline_strict": b.get("strict_correct"),
                "treatment_strict": t.get("strict_correct"),
            })
    return out


def _usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out = {}
    for system in sorted({str(row["system"]) for row in rows}):
        system_rows = [row for row in rows if str(row["system"]) == system]
        out[system] = {
            "events": len(system_rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in system_rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in system_rows),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in system_rows),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in system_rows),
            "avg_latency_s": round(mean(float(row.get("latency_s") or 0.0) for row in system_rows), 3),
        }
    return out


def _all_frozen_prereg(payloads: list[dict[str, Any]]) -> bool:
    for payload in payloads:
        path = payload.get("meta", {}).get("prereg")
        if not path:
            return False
        try:
            prereg = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return False
        if prereg.get("status") != "frozen":
            return False
        if prereg.get("algorithm", {}).get("id") != payload.get("meta", {}).get("algorithm_id"):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze ShardRCA external benchmark result files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..evaluation.baseline_selection import resolve_baseline
from ..evaluation.stats import aggregate_ci
from ..evaluation.stats import paired_bootstrap_effect, paired_mcnemar


def analyze_results(path: str | Path, *, baseline: str = "strongest_single", treatment: str = "rcaeval_shardrca_full") -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    rows = data.get("rows", [])
    paired_rows = [
        {
            **row,
            "case_id": str(row.get("case_id")),
            "system": str(row.get("system")),
            "strict_correct": bool(row.get("hit_at_1")),
            "score": float(row.get("mrr") or 0.0),
            "total_tokens": int(row.get("total_tokens") or 0),
        }
        for row in rows
    ]
    baseline, baseline_selection = resolve_baseline(paired_rows, requested=baseline, treatment=treatment)
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["system"])] = row

    disagreements = []
    for case_id, systems in sorted(by_case.items()):
        if baseline not in systems or treatment not in systems:
            continue
        b = systems[baseline]
        t = systems[treatment]
        if b.get("predicted_root") != t.get("predicted_root") or b.get("hit_at_1") != t.get("hit_at_1"):
            disagreements.append({
                "case_id": case_id,
                "baseline_root": b.get("predicted_root"),
                "treatment_root": t.get("predicted_root"),
                "true_root": t.get("true_root"),
                "baseline_hit_at_1": b.get("hit_at_1"),
                "treatment_hit_at_1": t.get("hit_at_1"),
            })

    paired = {}
    for metric in ("hit_at_1", "hit_at_3", "mrr", "fault_accuracy"):
        paired[metric] = {
            "mcnemar": paired_mcnemar(rows, baseline, treatment, metric),
            "effect": paired_bootstrap_effect(rows, baseline, treatment, metric),
        }
    summary = aggregate_ci(paired_rows, ["hit_at_1", "hit_at_3", "mrr", "fault_accuracy"])

    systems = sorted({row["system"] for row in rows})
    usage = {}
    for system in systems:
        sr = [row for row in rows if row["system"] == system]
        usage[system] = {
            "n": len(sr),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in sr),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in sr),
            "tool_calls": sum(int(row.get("tool_calls") or 0) for row in sr),
            "avg_latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in sr) / len(sr), 3) if sr else 0.0,
        }

    return {
        "benchmark": "rcaeval",
        "path": str(path),
        "systems": systems,
        "baseline": baseline,
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "summary": data.get("summary", {}),
        "computed_summary": summary,
        "paired": paired,
        "clear_win_gate": _clear_win_gate(summary, paired, baseline, treatment),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:25],
        "usage": usage,
        "limitations": [
            "RCAEval is operational/microservice evidence, not telecom evidence.",
            "Use as appendix/supporting evidence unless paired with real telecom or official telco fallback data.",
            "The headline comparison must use --baseline strongest_single or an explicitly justified stronger single.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze external RCA benchmark results")
    parser.add_argument("path")
    parser.add_argument("--baseline", default="strongest_single")
    parser.add_argument("--treatment", default="rcaeval_shardrca_full")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_results(args.path, baseline=args.baseline, treatment=args.treatment), indent=2))
    return 0


def _clear_win_gate(
    summary: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    baseline_acc = float(summary.get(baseline, {}).get("hit_at_1") or 0.0)
    treatment_acc = float(summary.get(treatment, {}).get("hit_at_1") or 0.0)
    absolute_delta = treatment_acc - baseline_acc
    error_reduction = absolute_delta / (1.0 - baseline_acc) if baseline_acc < 1.0 else 0.0
    p_value = paired["hit_at_1"]["mcnemar"]["p_value_exact"]
    effect_ok = absolute_delta >= 0.10 or error_reduction >= 0.20
    return {
        "passed": bool(effect_ok and p_value <= 0.05),
        "required": (
            "treatment beats strongest single baseline by >=0.10 absolute Hit@1 "
            "or >=20% relative error reduction, with exact paired p <= 0.05"
        ),
        "baseline": baseline,
        "treatment": treatment,
        "baseline_hit_at_1": round(baseline_acc, 4),
        "treatment_hit_at_1": round(treatment_acc, 4),
        "absolute_delta": round(absolute_delta, 4),
        "relative_error_reduction": round(error_reduction, 4),
        "observed_delta": round(absolute_delta, 4),
        "observed_p": p_value,
    }


if __name__ == "__main__":
    raise SystemExit(main())

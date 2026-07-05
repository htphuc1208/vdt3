"""Paired analysis for TeleLogsAgent profile, staged LLM, or HTTP-tool result files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..evaluation.baseline_selection import resolve_baseline
from ..evaluation.stats import aggregate_ci, paired_bootstrap_effect, paired_mcnemar


def analyze_results(
    paths: list[str | Path],
    *,
    baseline: str = "strongest_single",
    treatment: str = "shardrca_full",
) -> dict[str, Any]:
    payloads = _load_payloads(paths)
    rows = _rows_from_payloads(payloads)
    modes = _result_modes(payloads)
    paired_rows = [
        {
            "case_id": row["case_id"],
            "system": row["system"],
            "strict_correct": bool(row.get("strict_correct")),
            "score": float(row.get("score") or 0.0),
            "score_available": bool(row.get("score_available")),
            "total_tokens": int(row.get("total_tokens") or 0),
            "llm_calls": int(row.get("llm_calls") or 0),
            "tool_calls": int(row.get("tool_calls") or 0),
            "tool_failures": int(row.get("tool_failures") or 0),
            "tool_failure_rate": float(row.get("tool_failure_rate") or 0.0),
            "tool_call_efficiency": float(row.get("tool_call_efficiency") or 0.0),
            "latency_s": float(row.get("latency_s") or 0.0),
        }
        for row in rows
    ]
    baseline, baseline_selection = resolve_baseline(paired_rows, requested=baseline, treatment=treatment)
    summary = aggregate_ci(
        paired_rows,
        [
            "strict_correct",
            "score",
            "score_available",
            "tool_calls",
            "tool_failures",
            "tool_failure_rate",
            "tool_call_efficiency",
        ],
    )
    strict = {
        "mcnemar": paired_mcnemar(paired_rows, baseline, treatment, "strict_correct"),
        "effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "strict_correct"),
    }
    score = {
        "effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "score"),
    }
    return {
        "benchmark": "telelogs_agent",
        "paths": [str(path) for path in paths],
        "modes": modes,
        "official_tool_mode": modes == ["tool"],
        "evidence_mode": "official_http_tool" if modes == ["tool"] else "staged_or_mixed",
        "systems": sorted({row["system"] for row in rows}),
        "baseline": baseline,
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "summary": summary,
        "paired": {"strict_correct": strict, "score": score},
        "clear_win_gate": _clear_win_gate(summary, strict, baseline, treatment),
        "disagreements": _disagreements(rows, baseline, treatment),
        "usage": _usage(rows),
        "limitations": [
            "TeleLogsAgent is synthetic 5G fallback evidence, not real operator telemetry.",
            "Profile/LLM staged modes are not the official HTTP-tool benchmark path.",
            "Rows without evaluator labels are score_available=false and must not be counted as wins.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze staged TeleLogsAgent result files.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--baseline", default="strongest_single")
    parser.add_argument("--treatment", default="shardrca_full")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    payload = analyze_results(args.paths, baseline=args.baseline, treatment=args.treatment)
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
    return _rows_from_payloads(_load_payloads(paths))


def _load_payloads(paths: list[str | Path]) -> list[dict[str, Any]]:
    return [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in paths
    ]


def _rows_from_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for row in payload.get("rows", []):
            item = dict(row)
            if "case_id" not in item:
                item["case_id"] = f"{item.get('scenario_set')}:{item.get('row_id')}"
            rows.append(item)
    return rows


def _result_modes(payloads: list[dict[str, Any]]) -> list[str]:
    modes = {
        str((payload.get("meta") or {}).get("mode") or "unknown")
        for payload in payloads
    }
    return sorted(modes)


def _disagreements(rows: list[dict[str, Any]], baseline: str, treatment: str) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row["case_id"])][str(row["system"])] = row
    out = []
    for case_id, systems in sorted(by_case.items()):
        if baseline not in systems or treatment not in systems:
            continue
        b = systems[baseline]
        t = systems[treatment]
        if b.get("strict_correct") != t.get("strict_correct") or b.get("score") != t.get("score"):
            out.append({
                "case_id": case_id,
                "baseline_score": b.get("score"),
                "treatment_score": t.get("score"),
                "baseline_strict": b.get("strict_correct"),
                "treatment_strict": t.get("strict_correct"),
                "baseline_prediction": b.get("prediction"),
                "treatment_prediction": t.get("prediction"),
            })
    return out[:50]


def _usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for system in sorted({str(row["system"]) for row in rows}):
        sr = [row for row in rows if str(row["system"]) == system]
        out[system] = {
            "n": len(sr),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in sr),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in sr),
            "tool_calls": sum(int(row.get("tool_calls") or 0) for row in sr),
            "tool_failures": sum(int(row.get("tool_failures") or 0) for row in sr),
            "tool_failure_rate": round(
                sum(int(row.get("tool_failures") or 0) for row in sr)
                / max(1, sum(int(row.get("tool_calls") or 0) for row in sr)),
                4,
            ),
            "avg_tool_call_efficiency": round(
                sum(float(row.get("tool_call_efficiency") or 0.0) for row in sr) / len(sr),
                4,
            ) if sr else 0.0,
            "avg_latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in sr) / len(sr), 3) if sr else 0.0,
        }
    return out


def _clear_win_gate(
    summary: dict[str, dict[str, Any]],
    strict: dict[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    baseline_acc = float(summary.get(baseline, {}).get("strict_correct") or 0.0)
    treatment_acc = float(summary.get(treatment, {}).get("strict_correct") or 0.0)
    absolute_delta = treatment_acc - baseline_acc
    error_reduction = absolute_delta / (1.0 - baseline_acc) if baseline_acc < 1.0 else 0.0
    p_value = strict["mcnemar"]["p_value_exact"]
    effect_ok = absolute_delta >= 0.10 or error_reduction >= 0.20
    return {
        "passed": bool(effect_ok and p_value <= 0.05),
        "required": (
            "treatment beats strongest single baseline by >=0.10 absolute strict accuracy "
            "or >=20% relative error reduction, with exact paired p <= 0.05"
        ),
        "baseline": baseline,
        "treatment": treatment,
        "baseline_strict_accuracy": round(baseline_acc, 4),
        "treatment_strict_accuracy": round(treatment_acc, 4),
        "absolute_delta": round(absolute_delta, 4),
        "relative_error_reduction": round(error_reduction, 4),
        "observed_delta": round(absolute_delta, 4),
        "observed_p": p_value,
    }


if __name__ == "__main__":
    raise SystemExit(main())

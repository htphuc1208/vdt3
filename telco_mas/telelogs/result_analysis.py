"""Paired analysis for official TeleLogs result files."""
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
    rows = _load_rows(paths)
    paired_rows = [
        {
            "case_id": row["case_id"],
            "system": row["system"],
            "strict_correct": bool(row.get("strict_correct")),
            "score": float(row.get("score") or 0.0),
            "score_available": bool(row.get("score_available")),
            "total_tokens": int(row.get("total_tokens") or 0),
            "llm_calls": int(row.get("llm_calls") or 0),
            "latency_s": float(row.get("latency_s") or 0.0),
        }
        for row in rows
    ]
    baseline, baseline_selection = resolve_baseline(paired_rows, requested=baseline, treatment=treatment)
    summary = aggregate_ci(paired_rows, ["strict_correct", "score", "score_available"])
    paired = {
        "strict_correct": {
            "mcnemar": paired_mcnemar(paired_rows, baseline, treatment, "strict_correct"),
            "effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "strict_correct"),
        },
        "score": {"effect": paired_bootstrap_effect(paired_rows, baseline, treatment, "score")},
    }
    disagreements = _disagreements(rows, baseline, treatment)
    return {
        "benchmark": "telelogs",
        "paths": [str(path) for path in paths],
        "systems": sorted({row["system"] for row in rows}),
        "baseline": baseline,
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "summary": summary,
        "paired": paired,
        "clear_win_gate": _clear_win_gate(summary, paired, baseline, treatment),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:50],
        "usage": _usage(rows),
        "limitations": [
            "TeleLogs is official synthetic 5G fallback evidence, not real operator alarm telemetry.",
            "Strict correctness requires exact structured root cause set matching.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paired TeleLogs result files.")
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
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            item = dict(row)
            if "case_id" not in item:
                item["case_id"] = f"{item.get('split')}:{item.get('row_id')}"
            rows.append(item)
    return rows


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
    return out


def _usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for system in sorted({str(row["system"]) for row in rows}):
        sr = [row for row in rows if str(row["system"]) == system]
        out[system] = {
            "n": len(sr),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in sr),
            "llm_calls": sum(int(row.get("llm_calls") or 0) for row in sr),
            "avg_latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in sr) / len(sr), 3) if sr else 0.0,
        }
    return out


def _clear_win_gate(
    summary: dict[str, dict[str, Any]],
    paired: dict[str, Any],
    baseline: str,
    treatment: str,
) -> dict[str, Any]:
    baseline_acc = float(summary.get(baseline, {}).get("strict_correct") or 0.0)
    treatment_acc = float(summary.get(treatment, {}).get("strict_correct") or 0.0)
    absolute_delta = treatment_acc - baseline_acc
    error_reduction = absolute_delta / (1.0 - baseline_acc) if baseline_acc < 1.0 else 0.0
    p_value = paired["strict_correct"]["mcnemar"]["p_value_exact"]
    effect_ok = absolute_delta >= 0.10 or error_reduction >= 0.20
    return {
        "passed": bool(effect_ok and p_value <= 0.05),
        "required": "treatment-baseline strict delta >=0.10 or error reduction >=20%, exact paired p <=0.05",
        "baseline": baseline,
        "treatment": treatment,
        "baseline_strict_accuracy": round(baseline_acc, 4),
        "treatment_strict_accuracy": round(treatment_acc, 4),
        "absolute_delta": round(absolute_delta, 4),
        "relative_error_reduction": round(error_reduction, 4),
        "observed_p": p_value,
    }


if __name__ == "__main__":
    raise SystemExit(main())

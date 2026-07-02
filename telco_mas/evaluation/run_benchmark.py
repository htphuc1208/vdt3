"""Benchmark TelcoMAS on telecom and external RCA suites.

Usage:
    python -m telco_mas.evaluation.run_benchmark                 # all scenarios, both systems
    python -m telco_mas.evaluation.run_benchmark --scenarios fiber_cut,dns_failure
    python -m telco_mas.evaluation.run_benchmark --systems multi --no-cache
    python -m telco_mas.evaluation.run_benchmark --suite rcaeval --sample 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from ..config import get_settings
from ..environment.scenarios import get_scenario, list_scenario_ids
from ..llm import LLMClient, LLMError
from ..pipeline import run
from .metrics import aggregate, score_result
from .rcaeval_adapter import heuristic_predict, load_cases, score_predictions, validate_rcaeval
from .stats import aggregate_ci, paired_mcnemar

SYSTEM_MODES = {
    "multi": "multi",
    "full": "multi",
    "single": "single",
    "no_rag": "multi",
    "no_consensus": "multi",
    "no_arbiter": "multi",
}
RESEARCH_SYSTEMS = {"full", "single", "no_rag", "no_consensus", "no_arbiter"}


def _print_summary(summary: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Benchmark summary (accuracy %, efficiency avg)")
        table.add_column("System")
        for col in ["Local.", "RootCause", "Diagnosis", "Resolved", "Tokens", "ToolCalls", "LLMCalls", "Latency"]:
            table.add_column(col, justify="right")
        for system, s in summary.items():
            table.add_row(
                system,
                f"{s['localization_accuracy']*100:.0f}",
                f"{s['root_cause_accuracy']*100:.0f}",
                f"{s['diagnosis_accuracy']*100:.0f}",
                f"{s['resolution_rate']*100:.0f}",
                f"{s['avg_total_tokens']:.0f}",
                f"{s['avg_tool_calls']:.0f}",
                f"{s['avg_llm_calls']:.0f}",
                f"{s['avg_latency_s']:.1f}s",
            )
        Console().print(table)
    except Exception:
        print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TelcoMAS benchmark")
    parser.add_argument("--suite", default="telco_v1", help="comma list: telco_v1,telco_v2,rcaeval,openrca,kaggle_supplement")
    parser.add_argument("--scenarios", default="all", help="comma list of scenario ids, or 'all'")
    parser.add_argument("--systems", default="full,single", help="comma list: full,single,no_rag,no_consensus,no_arbiter")
    parser.add_argument("--sample", type=int, default=0, help="sample size for external suites")
    parser.add_argument("--runs", type=int, default=1, help="repeat count for stochastic LLM experiments")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rcaeval-root", default=None)
    parser.add_argument("--no-cache", action="store_true", help="disable the LLM response cache")
    parser.add_argument("--cache-only", action="store_true",
                        help="replay from cache only; never make a live LLM call (offline re-scoring)")
    parser.add_argument("--holdout-sop", action="store_true",
                        help="remove each scenario's matching SOP + historical incidents from the KB (construct-validity control)")
    parser.add_argument("--kb-distractors", action="store_true",
                        help="add off-target distractor SOPs/incidents to make retrieval non-trivial")
    parser.add_argument("--out", default="results/benchmark.json")
    parser.add_argument("--figures", default="report/figures")
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    if suites == ["rcaeval"]:
        return _run_rcaeval(args)
    if "openrca" in suites:
        print("OpenRCA is handled by `make bench-openrca` / `python -m telco_mas.openrca.cli`.", file=sys.stderr)
        return 2
    if "kaggle_supplement" in suites:
        print("Kaggle supplement requires external credentials and is documented, but not a headline benchmark.", file=sys.stderr)
        return 2

    settings = get_settings()
    if not settings.has_api_key:
        print(
            "ERROR: no OPENAI_API_KEY set. The benchmark runs live LLM agents.\n"
            "Set OPENAI_API_KEY (OpenAI) or point OPENAI_BASE_URL at DeepSeek with its key. "
            "See .env.example.",
            file=sys.stderr,
        )
        return 2

    scenario_ids = list_scenario_ids() if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    requested_systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEM_MODES]
    if args.runs > 1 and not (args.no_cache or args.cache_only):
        print("WARNING: --runs>1 with the cache ON returns identical outputs for every run "
              "(same cache key). Add --no-cache to measure LLM variance.", file=sys.stderr)
    llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    rows: list[dict] = []
    for run_idx in range(max(1, args.runs)):
        for sid in scenario_ids:
            scenario = get_scenario(sid)
            for requested in requested_systems:
                label = "single_agent" if requested == "single" else "multi_agent"
                print(f"  running [{requested}:{label}] on scenario '{sid}' run={run_idx + 1} …", flush=True)
                try:
                    result = run(scenario, mode=requested, llm=llm,
                                 holdout=args.holdout_sop, kb_distractors=args.kb_distractors)
                except LLMError as exc:
                    print(f"    LLM error: {exc}", file=sys.stderr)
                    return 1
                score = score_result(result, scenario)
                score["system"] = requested
                score["run"] = run_idx
                rows.append(score)
                mark = "OK " if score["diagnosis_correct"] else "MISS"
                print(f"    -> [{mark}] pred={score['predicted_element']} true={score['true_element']} "
                      f"resolved={score['resolved']} tokens={score['total_tokens']}")

    summary = aggregate(rows)
    summary_ci = aggregate_ci(rows, ["localization", "root_cause_correct", "diagnosis_correct", "resolved"])
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": settings.model,
            "provider": settings.provider_label,
            "suite": suites,
            "scenarios": scenario_ids,
            "systems": requested_systems,
            "runs": args.runs,
            "seed": args.seed,
            "cache": (not args.no_cache),
            "cache_only": args.cache_only,
            "holdout_sop": args.holdout_sop,
            "kb_distractors": args.kb_distractors,
            "fault_type_metric": "semantic-family-match",
        },
        "summary": summary,
        "summary_ci": summary_ci,
        "paired_tests": [
            paired_mcnemar(rows, "single", "full", "diagnosis_correct")
        ] if {"single", "full"}.issubset(set(requested_systems)) else [],
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved results -> {args.out}")

    if len(summary) >= 1:
        try:
            from .plots import make_charts

            charts = make_charts(summary, outdir=args.figures)
            print("Saved charts -> " + ", ".join(charts))
        except Exception as exc:
            print(f"Skipped charts: {exc}")

    _print_summary(summary)
    return 0


def _run_rcaeval(args: argparse.Namespace) -> int:
    validation = validate_rcaeval(args.rcaeval_root)
    if not validation["ok"]:
        print(f"RCAEval validation failed: {json.dumps(validation, indent=2)}", file=sys.stderr)
        return 2
    systems = [s.strip() for s in args.systems.split(",") if s.strip() in RESEARCH_SYSTEMS]
    if not systems:
        systems = ["full"]
    cases = load_cases(args.rcaeval_root, sample=args.sample or None, seed=args.seed)
    predictions = []
    for system in systems:
        for case in cases:
            pred = heuristic_predict(case, system=f"rcaeval_{system}")
            # Conservative ablation smoke behavior: variants share the same
            # label-safe profile features until LLM-agent RCAEval is run.
            predictions.append(pred)
    scored = score_predictions(cases, predictions)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "rcaeval",
            "systems": systems,
            "sample": args.sample or len(cases),
            "seed": args.seed,
            "validation": validation,
            "mode": "label-safe RCAEval profile smoke benchmark",
        },
        **scored,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved RCAEval results -> {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import time
from datetime import datetime

from ..config import get_settings
from ..environment.scenarios import get_scenario, list_scenario_ids
from ..llm import LLMClient, LLMError, extract_json
from ..pipeline import run
from .external import ExternalBenchmarkCase, ExternalPrediction
from .metrics import aggregate, score_result
from .rcaeval_adapter import heuristic_predict, load_cases, score_predictions, validate_rcaeval
from .stats import aggregate_ci, paired_bootstrap_effect, paired_mcnemar

SYSTEM_MODES = {
    "multi": "multi",
    "full": "multi",
    "single": "single",
    "no_rag": "multi",
    "no_consensus": "multi",
    "no_arbiter": "multi",
    "no_partition": "multi",
    "no_debate": "multi",
}
RESEARCH_SYSTEMS = {
    "full", "single", "no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate"
}


def _print_summary(summary: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Benchmark summary (accuracy %, efficiency avg)")
        table.add_column("System")
        for col in [
            "Local.", "FaultType", "Causal", "Diagnosis", "E2E", "Resolved",
            "RemTarget", "RemAction", "Debate", "Tokens", "Solved/10k", "ToolCalls", "LLMCalls", "Latency",
        ]:
            table.add_column(col, justify="right")
        for system, s in summary.items():
            table.add_row(
                system,
                f"{s['localization_accuracy']*100:.0f}",
                f"{s['fault_type_accuracy']*100:.0f}",
                f"{s['causal_explanation_accuracy']*100:.0f}",
                f"{s['diagnosis_accuracy']*100:.0f}",
                f"{s['end_to_end_accuracy']*100:.0f}",
                f"{s['resolution_rate']*100:.0f}",
                f"{s['remediation_target_accuracy']*100:.0f}",
                f"{s['remediation_action_accuracy']*100:.0f}",
                f"{s.get('debate_call_rate', 0.0)*100:.0f}",
                f"{s['avg_total_tokens']:.0f}",
                f"{s['solved_cases_per_10k_tokens']:.3f}",
                f"{s['avg_tool_calls']:.0f}",
                f"{s['avg_llm_calls']:.0f}",
                f"{s['avg_latency_s']:.1f}s",
            )
        Console().print(table)
    except Exception:
        print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TelcoMAS benchmark")
    parser.add_argument("--suite", default="telco_v1", help="comma list: telco_v1,telco_v2,telco_v3,rcaeval,openrca,kaggle_supplement")
    parser.add_argument("--scenarios", default="all", help="comma list of scenario ids, or 'all'")
    parser.add_argument("--systems", default="full,single", help="comma list: full,single,no_rag,no_consensus,no_arbiter,no_partition,no_debate")
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
    parser.add_argument("--out", default=None)
    parser.add_argument("--figures", default="report/figures")
    parser.add_argument("--external-mode", choices=["profile", "llm"], default="profile",
                        help="external suite inference mode; profile is a smoke heuristic, llm makes live model calls")
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    unknown_suites = set(suites) - {"telco_v1", "telco_v2", "telco_v3", "rcaeval", "openrca", "kaggle_supplement"}
    if unknown_suites:
        print(f"Unknown suite(s): {', '.join(sorted(unknown_suites))}", file=sys.stderr)
        return 2
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

    telco_suites = [s for s in suites if s in {"telco_v1", "telco_v2", "telco_v3"}] or ["telco_v1"]
    scenario_ids = _scenario_ids(args.scenarios, telco_suites)
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
                    holdout = args.holdout_sop or ("no_exact_sop" in scenario.stress_tags)
                    result = run(scenario, mode=requested, llm=llm,
                                 holdout=holdout, kb_distractors=args.kb_distractors)
                except LLMError as exc:
                    print(f"    LLM error: {exc}", file=sys.stderr)
                    return 1
                score = score_result(result, scenario)
                score["system"] = requested
                score["run"] = run_idx
                score["kb_holdout_effective"] = holdout
                rows.append(score)
                mark = "OK " if score["diagnosis_correct"] else "MISS"
                print(f"    -> [{mark}] pred={score['predicted_element']} true={score['true_element']} "
                      f"resolved={score['resolved']} tokens={score['total_tokens']}")

    summary = aggregate(rows)
    summary_ci = aggregate_ci(rows, [
        "localization",
        "fault_type_correct",
        "causal_explanation_correct",
        "diagnosis_correct",
        "end_to_end_correct",
        "remediation_target_correct",
        "remediation_action_correct",
        "resolved",
    ])
    paired_tests = []
    paired_effects = []
    if {"single", "full"}.issubset(set(requested_systems)):
        paired_tests = [
            paired_mcnemar(rows, "single", "full", "diagnosis_correct"),
            paired_mcnemar(rows, "single", "full", "end_to_end_correct"),
        ]
        paired_effects = [
            paired_bootstrap_effect(rows, "single", "full", "diagnosis_correct"),
            paired_bootstrap_effect(rows, "single", "full", "end_to_end_correct"),
        ]
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
            "metric_definitions": {
                "root_cause_correct": "loose root-cause keyword match (legacy secondary metric)",
                "fault_type_correct": "semantic fault-family match",
                "causal_explanation_correct": "at least two causal keywords or the maximum available, with >=0.30 recall",
                "diagnosis_correct": "strict: localization + fault_type_correct + causal_explanation_correct",
                "end_to_end_correct": "strict diagnosis and simulator-resolved validation",
                "solved_cases_per_10k_tokens": "end_to_end_correct cases divided by total tokens, scaled by 10k",
            },
        },
        "summary": summary,
        "summary_ci": summary_ci,
        "paired_tests": paired_tests,
        "paired_effects": paired_effects,
        "rows": rows,
    }
    args.out = args.out or _default_output_path(args, telco_suites, requested_systems)
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
    llm = None
    if args.external_mode == "llm":
        settings = get_settings()
        if not settings.has_api_key:
            print("ERROR: --external-mode llm requires OPENAI_API_KEY.", file=sys.stderr)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)
    for system in systems:
        for case in cases:
            if llm is None:
                pred = heuristic_predict(case, system=f"rcaeval_{system}")
                # Conservative smoke behavior: variants share the same
                # label-safe profile features and are not ablation evidence.
            else:
                pred = _llm_predict_rcaeval(case, system=f"rcaeval_{system}", llm=llm)
            predictions.append(pred)
    scored = score_predictions(cases, predictions)
    args.out = args.out or _default_output_path(args, ["rcaeval"], systems)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "rcaeval",
            "systems": systems,
            "sample": args.sample or len(cases),
            "seed": args.seed,
            "validation": validation,
            "mode": (
                "label-safe RCAEval profile smoke benchmark"
                if args.external_mode == "profile"
                else "label-safe RCAEval live LLM benchmark"
            ),
            "evidence_warning": (
                "profile mode is an ingestion/scoring smoke test and must not be used "
                "as multi-agent or ablation evidence"
            ) if args.external_mode == "profile" else "",
            "llm_mode_warning": (
                "LLM mode uses label-safe summarized telemetry. It is a live external-data "
                "baseline, not a faithful TelcoMAS tool-agent ablation."
            ) if args.external_mode == "llm" else "",
        },
        **scored,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved RCAEval results -> {args.out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _llm_predict_rcaeval(case: ExternalBenchmarkCase, *, system: str, llm: LLMClient) -> ExternalPrediction:
    system_prompt = """You are evaluating a root-cause analysis case from RCAEval.
Use only the label-safe telemetry summary. Do not assume access to hidden labels.
Return ONLY JSON:
{"root": "service_name", "ranked_roots": ["service1", "service2", "service3"],
 "fault_type": "cpu|mem|disk|delay|loss|socket|unknown",
 "confidence": 0.0, "notes": "short evidence summary"}"""
    user_prompt = json.dumps(case.inference_payload(), indent=2)
    started = time.time()
    resp = llm.chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        force_json=True,
    )
    data = extract_json(resp.content)
    ranked = data.get("ranked_roots")
    if not isinstance(ranked, list):
        ranked = []
    root = str(data.get("root") or (ranked[0] if ranked else "UNKNOWN"))
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return ExternalPrediction(
        case_id=case.case_id,
        system=system,
        root=root,
        ranked_roots=[str(item) for item in ranked] or [root],
        fault_type=str(data.get("fault_type") or "unknown"),
        accepted=root != "UNKNOWN",
        confidence=confidence,
        latency_s=round(time.time() - started, 2),
        total_tokens=resp.usage.total_tokens,
        tool_calls=resp.usage.tool_calls,
        llm_calls=resp.usage.llm_calls,
        notes=str(data.get("notes") or "live LLM prediction over label-safe RCAEval summary"),
    )


def _scenario_ids(scenario_arg: str, suites: list[str]) -> list[str]:
    if scenario_arg != "all":
        return [s.strip() for s in scenario_arg.split(",") if s.strip()]
    out: list[str] = []
    for suite in suites:
        out.extend(list_scenario_ids(suite=suite))
    return out


def _default_output_path(args: argparse.Namespace, suites: list[str], systems: list[str]) -> str:
    suite_label = "_".join(suites)
    system_set = set(systems)
    if suite_label == "rcaeval":
        suffix = "llm" if getattr(args, "external_mode", "profile") == "llm" else "profile"
        sample = f"_sample{args.sample}" if args.sample else ""
        return f"results/rcaeval_{suffix}{sample}.json"
    if args.holdout_sop or args.kb_distractors:
        parts = ["construct"]
        if args.holdout_sop:
            parts.append("holdout")
        if args.kb_distractors:
            parts.append("distractors")
        return f"results/{'_'.join(parts)}_{suite_label}.json"
    if {"no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate"} & system_set:
        return f"results/ablation_{suite_label}_runs{args.runs}.json"
    return f"results/benchmark_{suite_label}.json"


if __name__ == "__main__":
    raise SystemExit(main())

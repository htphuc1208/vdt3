"""Benchmark ShardRCA and related RCA baselines.

Usage:
    python -m telco_mas.evaluation.run_benchmark --suite rcaeval --sample 30
    python -m telco_mas.evaluation.run_benchmark --suite rcaeval_hard_llm --sample 20 --systems shardrca_full,single_react_sc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from ..config import get_settings
from ..environment.scenarios import get_scenario, list_scenario_ids
from ..llm import LLMClient, LLMError, extract_json
from ..pipeline import run
from ..shardrca.hard_split import load_hard_cases
from ..shardrca.runner import SHARDRCA_SYSTEMS, SINGLE_SYSTEMS, run_rcaeval_case
from ..synthetic_telco.dataset import load_scenarios, validate_dataset
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
    "no_verifier": "multi",
    "no_repair": "multi",
}
RESEARCH_SYSTEMS = {
    "full", "single", "single_sc", "no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate",
    "no_verifier", "no_repair",
    "shardrca_full", "shardrca_local_fusion", "no_falsifier", "no_vote", "no_shard",
    "single_react", "single_react_sc", "single_equal_tokens", "code_retrieval_single", "same_board_single",
}


def _print_summary(summary: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Benchmark summary (accuracy %, efficiency avg)")
        table.add_column("System")
        for col in [
            "Local.", "FaultType", "Causal", "Diagnosis", "E2E", "Resolved",
            "RemTarget", "RemAction", "Repair", "Debate", "Tokens", "Solved/10k", "ToolCalls", "LLMCalls", "Latency",
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
                f"{s.get('avg_remediation_attempts', 0.0):.1f}",
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
    parser = argparse.ArgumentParser(description="ShardRCA benchmark")
    parser.add_argument("--suite", default="rcaeval_hard", help="comma list: telco_v1,telco_v2,telco_v3,telco_v4,rcaeval,rcaeval_llm,rcaeval_hard,rcaeval_hard_llm,openrca,openrca_llm,kaggle_supplement")
    parser.add_argument("--scenarios", default="all", help="comma list of scenario ids, or 'all'")
    parser.add_argument("--systems", default="shardrca_full,single_react_sc,same_board_single", help="comma list: full,single,single_sc,single_react,single_react_sc,single_equal_tokens,code_retrieval_single,same_board_single,shardrca_full,shardrca_local_fusion,no_falsifier,no_vote,no_shard,no_rag,no_consensus,no_arbiter,no_partition,no_debate,no_verifier,no_repair")
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
    parser.add_argument("--checkpoint-dir", default=None,
                        help="per-case JSON checkpoint directory for external LLM runs")
    parser.add_argument("--synthetic-dataset", default=None,
                        help="frozen telco_v4 dataset artifact; required for telco_v4")
    parser.add_argument("--preregistration", default=None,
                        help="frozen preregistration to verify before a confirmatory telco_v4 run")
    parser.add_argument("--algorithm-id", default=None,
                        help="algorithm artifact ID; must equal the telco_v4 preregistration")
    parser.add_argument("--external-mode", choices=["profile", "llm"], default="profile",
                        help="external suite inference mode; profile is a smoke heuristic, llm makes live model calls")
    args = parser.parse_args(argv)

    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    unknown_suites = set(suites) - {"telco_v1", "telco_v2", "telco_v3", "telco_v4", "rcaeval", "rcaeval_llm", "rcaeval_hard", "rcaeval_hard_llm", "openrca", "openrca_llm", "kaggle_supplement"}
    if unknown_suites:
        print(f"Unknown suite(s): {', '.join(sorted(unknown_suites))}", file=sys.stderr)
        return 2
    if suites == ["rcaeval_llm"]:
        args.external_mode = "llm"
        return _run_rcaeval(args)
    if suites == ["rcaeval_hard_llm"]:
        args.external_mode = "llm"
        args.rcaeval_hard = True
        return _run_rcaeval(args)
    if suites == ["rcaeval_hard"]:
        args.rcaeval_hard = True
        return _run_rcaeval(args)
    if suites == ["rcaeval"]:
        return _run_rcaeval(args)
    if suites == ["openrca_llm"]:
        return _run_openrca_llm(args)
    if "openrca" in suites:
        print("OpenRCA is handled by `make bench-openrca` / `python -m telco_mas.openrca.cli`.", file=sys.stderr)
        return 2
    if "kaggle_supplement" in suites:
        print("Kaggle supplement requires external credentials and is documented, but not a headline benchmark.", file=sys.stderr)
        return 2

    frozen_scenarios: dict[str, object] | None = None
    frozen_dataset_meta: dict | None = None
    if "telco_v4" in suites:
        if (
            suites != ["telco_v4"]
            or not args.synthetic_dataset
            or not args.preregistration
            or not args.algorithm_id
        ):
            print(
                "ERROR: telco_v4 must run alone with --synthetic-dataset, "
                "--preregistration, and the matching --algorithm-id.",
                file=sys.stderr,
            )
            return 2
        try:
            dataset_path = Path(args.synthetic_dataset)
            dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            validation = validate_dataset(dataset_payload)
            if not validation["ok"]:
                raise ValueError("; ".join(validation["errors"]))
            if dataset_payload.get("meta", {}).get("suite") != "telco_v4":
                raise ValueError("dataset meta.suite is not telco_v4")
            loaded = load_scenarios(dataset_payload, verify_runtime=True)
            frozen_scenarios = {scenario.id: scenario for scenario in loaded}
            frozen_dataset_meta = {
                "path": str(dataset_path),
                "sha256": _sha256_file(dataset_path),
                "content_sha256": dataset_payload.get("meta", {}).get("content_sha256"),
                "design": dataset_payload.get("meta", {}).get("design"),
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: invalid frozen telco_v4 dataset: {exc}", file=sys.stderr)
            return 2
    elif args.synthetic_dataset or args.preregistration or args.algorithm_id:
        print(
            "ERROR: --synthetic-dataset/--preregistration/--algorithm-id are reserved for telco_v4.",
            file=sys.stderr,
        )
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

    telco_suites = [s for s in suites if s in {"telco_v1", "telco_v2", "telco_v3", "telco_v4"}] or ["telco_v1"]
    if frozen_scenarios is not None:
        scenario_ids = (
            list(frozen_scenarios)
            if args.scenarios == "all"
            else [item.strip() for item in args.scenarios.split(",") if item.strip()]
        )
        unknown = sorted(set(scenario_ids) - set(frozen_scenarios))
        if unknown:
            print(f"ERROR: scenario(s) absent from frozen artifact: {', '.join(unknown)}", file=sys.stderr)
            return 2
    else:
        scenario_ids = _scenario_ids(args.scenarios, telco_suites)
    requested_systems = [s.strip() for s in args.systems.split(",") if s.strip() in SYSTEM_MODES]
    if not requested_systems:
        print("ERROR: no runnable telco systems selected.", file=sys.stderr)
        return 2
    if args.preregistration:
        try:
            _verify_preregistration(
                Path(args.preregistration),
                dataset_meta=frozen_dataset_meta or {},
                scenario_ids=scenario_ids,
                systems=requested_systems,
                model=settings.model,
                temperature=settings.temperature,
                base_url=settings.base_url,
                max_tool_iters=settings.max_tool_iters,
                runs=args.runs,
                no_cache=args.no_cache,
                algorithm_id=args.algorithm_id,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: preregistration verification failed: {exc}", file=sys.stderr)
            return 2
    if args.runs > 1 and not (args.no_cache or args.cache_only):
        print("WARNING: --runs>1 with the cache ON returns identical outputs for every run "
              "(same cache key). Add --no-cache to measure LLM variance.", file=sys.stderr)
    llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    rows: list[dict] = []
    for run_idx in range(max(1, args.runs)):
        for sid in scenario_ids:
            scenario = frozen_scenarios[sid] if frozen_scenarios is not None else get_scenario(sid)
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
            "base_url": settings.base_url,
            "temperature": settings.temperature,
            "max_tool_iters": settings.max_tool_iters,
            "suite": suites,
            "scenarios": scenario_ids,
            "systems": requested_systems,
            "runs": args.runs,
            "seed": args.seed,
            "cache": (not args.no_cache),
            "cache_only": args.cache_only,
            "holdout_sop": args.holdout_sop,
            "kb_distractors": args.kb_distractors,
            "frozen_dataset": frozen_dataset_meta,
            "preregistration": args.preregistration,
            "algorithm_id": args.algorithm_id,
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
    hard = bool(getattr(args, "rcaeval_hard", False))
    cases = (
        load_hard_cases(args.rcaeval_root, sample=args.sample or None, seed=args.seed)
        if hard
        else load_cases(args.rcaeval_root, sample=args.sample or None, seed=args.seed)
    )
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
            checkpoint_suite = "rcaeval_hard_v7" if hard else "rcaeval_v7"
            checkpoint = _load_prediction_checkpoint(args, checkpoint_suite, system, case.case_id)
            if checkpoint is not None:
                pred = checkpoint
            elif llm is None:
                pred = heuristic_predict(case, system=f"rcaeval_{system}")
                # Conservative smoke behavior: variants share the same
                # label-safe profile features and are not ablation evidence.
            elif system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS:
                pred = run_rcaeval_case(case, system=system, llm=llm)
            else:
                pred = _llm_predict_rcaeval(case, system=f"rcaeval_{system}", llm=llm)
            _write_prediction_checkpoint(args, checkpoint_suite, system, case.case_id, pred)
            predictions.append(pred)
    scored = score_predictions(cases, predictions)
    suite_name = "rcaeval_hard" if hard else "rcaeval"
    args.out = args.out or _default_output_path(args, [suite_name], systems)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": suite_name,
            "systems": systems,
            "sample": args.sample or len(cases),
            "seed": args.seed,
            "validation": validation,
            "hard_split": hard,
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
                "baseline, not a faithful synthetic tool-agent ablation."
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


def _run_openrca_llm(args: argparse.Namespace) -> int:
    from ..openrca.cli import main as openrca_main

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    system = systems[0] if systems else "shardrca_full"
    argv = [
        "--mode", "llm",
        "--system", system,
        "--limit", str(args.sample or 3),
        "--out", args.out or _default_output_path(args, ["openrca_llm"], [system]),
    ]
    return openrca_main(argv)


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


def _verify_preregistration(
    path: Path,
    *,
    dataset_meta: dict,
    scenario_ids: list[str],
    systems: list[str],
    model: str,
    temperature: float,
    base_url: str,
    max_tool_iters: int,
    runs: int,
    no_cache: bool,
    algorithm_id: str,
) -> None:
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen":
        raise ValueError("status is not frozen")
    if prereg.get("dataset", {}).get("sha256") != dataset_meta.get("sha256"):
        raise ValueError("dataset SHA-256 differs from preregistration")
    expected_ids = prereg.get("row_selection", {}).get("source_scenario_ids", [])
    if expected_ids and expected_ids != scenario_ids:
        raise ValueError("scenario order/selection differs from preregistration")
    if prereg.get("systems") != systems:
        raise ValueError("system list/order differs from preregistration")
    if prereg.get("algorithm", {}).get("id") != algorithm_id:
        raise ValueError("algorithm ID differs from preregistration")
    expected_model = prereg.get("model", {}).get("name")
    if expected_model and expected_model != model:
        raise ValueError(f"model differs: expected {expected_model!r}, got {model!r}")
    expected_temperature = prereg.get("model", {}).get("temperature")
    if expected_temperature is not None and float(expected_temperature) != float(temperature):
        raise ValueError(
            f"temperature differs: expected {expected_temperature}, got {temperature}"
        )
    expected_base_url = prereg.get("model", {}).get("base_url")
    if expected_base_url and expected_base_url.rstrip("/") != base_url.rstrip("/"):
        raise ValueError("LLM base URL differs from preregistration")
    expected_tool_iters = prereg.get("model", {}).get("max_tool_iters")
    if expected_tool_iters is not None and int(expected_tool_iters) != max_tool_iters:
        raise ValueError("max tool iterations differ from preregistration")
    expected_runs = prereg.get("execution", {}).get("runs")
    if expected_runs is not None and expected_runs != runs:
        raise ValueError(f"run count differs: expected {expected_runs}, got {runs}")
    if prereg.get("model", {}).get("cache") is False and not no_cache:
        raise ValueError("preregistered run requires --no-cache")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output_path(args: argparse.Namespace, suites: list[str], systems: list[str]) -> str:
    suite_label = "_".join(suites)
    system_set = set(systems)
    if suite_label == "rcaeval":
        suffix = "llm" if getattr(args, "external_mode", "profile") == "llm" else "profile"
        sample = f"_sample{args.sample}" if args.sample else ""
        return f"results/rcaeval_{suffix}{sample}.json"
    if suite_label == "rcaeval_hard":
        suffix = "llm" if getattr(args, "external_mode", "profile") == "llm" else "profile"
        sample = f"_sample{args.sample}" if args.sample else ""
        systems_label = "_".join(systems) if systems else "default"
        return f"results/rcaeval_hard_{suffix}_{systems_label}{sample}.json"
    if suite_label == "openrca_llm":
        sample = f"_sample{args.sample}" if args.sample else ""
        systems_label = "_".join(systems) if systems else "shardrca_full"
        return f"results/openrca_llm_{systems_label}{sample}.json"
    if args.holdout_sop or args.kb_distractors:
        parts = ["construct"]
        if args.holdout_sop:
            parts.append("holdout")
        if args.kb_distractors:
            parts.append("distractors")
        return f"results/{'_'.join(parts)}_{suite_label}.json"
    if {"no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate", "no_verifier", "no_repair"} & system_set:
        return f"results/ablation_{suite_label}_runs{args.runs}.json"
    return f"results/benchmark_{suite_label}.json"


def _checkpoint_dir(args: argparse.Namespace, suite: str) -> str | None:
    if args.checkpoint_dir:
        return args.checkpoint_dir
    if getattr(args, "external_mode", "profile") == "llm":
        return f"results/checkpoints/{suite}_llm"
    return None


def _checkpoint_path(args: argparse.Namespace, suite: str, system: str, case_id: str) -> str | None:
    root = _checkpoint_dir(args, suite)
    if not root:
        return None
    safe_case = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)
    safe_system = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in system)
    return os.path.join(root, safe_system, f"{safe_case}.json")


def _load_prediction_checkpoint(args: argparse.Namespace, suite: str, system: str, case_id: str) -> ExternalPrediction | None:
    path = _checkpoint_path(args, suite, system, case_id)
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return ExternalPrediction(**json.load(fh))


def _write_prediction_checkpoint(
    args: argparse.Namespace,
    suite: str,
    system: str,
    case_id: str,
    prediction: ExternalPrediction,
) -> None:
    path = _checkpoint_path(args, suite, system, case_id)
    if not path or os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(prediction.__dict__, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

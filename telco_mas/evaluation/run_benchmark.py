"""Paper-facing benchmark dispatcher for ShardRCA.

Supported suites are intentionally narrow after the paper-only cleanup:
RCAEval/RCAEval-Hard and the OpenRCA runner. Non-paper entry points were removed
so this module matches the paper surface.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime

from ..config import get_settings
from ..llm import LLMClient, extract_json
from ..shardrca.hard_split import load_hard_cases
from ..shardrca.runner import SHARDRCA_SYSTEMS, SINGLE_SYSTEMS, run_rcaeval_case
from .external import ExternalBenchmarkCase, ExternalPrediction
from .rcaeval_adapter import (
    apply_rcaeval_metric_prior,
    heuristic_predict,
    load_cases,
    score_predictions,
    validate_rcaeval,
)

RESEARCH_SYSTEMS = {
    "shardrca_full",
    "shardrca_local_fusion",
    "no_falsifier",
    "no_vote",
    "no_shard",
    "no_interaction",
    "no_topology",
    "no_refinement",
    "single_react",
    "single_react_sc",
    "single_equal_tokens",
    "code_retrieval_single",
    "full",
    "single",
    "single_sc",
}

SUPPORTED_SUITES = {
    "rcaeval",
    "rcaeval_llm",
    "rcaeval_hard",
    "rcaeval_hard_llm",
    "openrca",
    "openrca_llm",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ShardRCA paper benchmark dispatcher")
    parser.add_argument(
        "--suite",
        default="rcaeval_hard",
        help="comma list: rcaeval,rcaeval_llm,rcaeval_hard,rcaeval_hard_llm,openrca,openrca_llm",
    )
    parser.add_argument(
        "--systems",
        default="shardrca_full,single_react_sc",
        help="comma list of ShardRCA/RCAEval systems",
    )
    parser.add_argument("--sample", type=int, default=0, help="sample size for external suites")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rcaeval-root", default=None)
    parser.add_argument(
        "--rcaeval-source",
        default=None,
        help="optional comma-separated RCAEval dataset filter, e.g. RE2-TT",
    )
    parser.add_argument(
        "--rcaeval-exclude-case-ids",
        default=None,
        help="optional comma-separated engineering case IDs excluded before sampling",
    )
    parser.add_argument("--case-concurrency", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true", help="disable the LLM response cache")
    parser.add_argument("--cache-only", action="store_true", help="replay from cache only; never make a live LLM call")
    parser.add_argument("--checkpoint-dir", default=None, help="per-case JSON checkpoint directory for external LLM runs")
    parser.add_argument("--external-mode", choices=["profile", "llm"], default="profile")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    suites = [suite.strip() for suite in args.suite.split(",") if suite.strip()]
    unknown = set(suites) - SUPPORTED_SUITES
    if unknown:
        print(f"Unknown suite(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    if len(suites) != 1:
        print("Run one paper suite at a time.", file=sys.stderr)
        return 2

    suite = suites[0]
    if suite == "rcaeval_llm":
        args.external_mode = "llm"
        args.rcaeval_hard = False
        return _run_rcaeval(args)
    if suite == "rcaeval_hard_llm":
        args.external_mode = "llm"
        args.rcaeval_hard = True
        return _run_rcaeval(args)
    if suite == "rcaeval_hard":
        args.rcaeval_hard = True
        return _run_rcaeval(args)
    if suite == "rcaeval":
        args.rcaeval_hard = False
        return _run_rcaeval(args)
    if suite in {"openrca", "openrca_llm"}:
        return _run_openrca(args, llm_mode=suite == "openrca_llm")
    return 2


def _run_rcaeval(args: argparse.Namespace) -> int:
    validation = validate_rcaeval(args.rcaeval_root)
    if not validation["ok"]:
        print(f"RCAEval validation failed: {json.dumps(validation, indent=2)}", file=sys.stderr)
        return 2

    systems = [system.strip() for system in args.systems.split(",") if system.strip() in RESEARCH_SYSTEMS]
    if not systems:
        systems = ["shardrca_full"]

    hard = bool(getattr(args, "rcaeval_hard", False))
    source_filter = {source.strip() for source in str(args.rcaeval_source or "").split(",") if source.strip()}
    excluded_case_ids = {
        case_id.strip()
        for case_id in str(args.rcaeval_exclude_case_ids or "").split(",")
        if case_id.strip()
    }
    cases = _load_rcaeval_cases(args, hard=hard, source_filter=source_filter, excluded_case_ids=excluded_case_ids)
    if not cases:
        print(f"RCAEval source filter selected no cases: {sorted(source_filter)}", file=sys.stderr)
        return 2

    llm = None
    if args.external_mode == "llm":
        settings = get_settings()
        if not settings.has_api_key:
            print("ERROR: --external-mode llm requires OPENAI_API_KEY.", file=sys.stderr)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    def infer(system: str, case: ExternalBenchmarkCase) -> ExternalPrediction:
        source_suffix = "_".join(sorted(source_filter)).lower() if source_filter else "all"
        checkpoint_suite = f"rcaeval_hard_v7_{source_suffix}" if hard else f"rcaeval_v7_{source_suffix}"
        checkpoint = _load_prediction_checkpoint(args, checkpoint_suite, system, case.case_id)
        if checkpoint is not None:
            prediction = checkpoint
            if system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS:
                prediction = apply_rcaeval_metric_prior(case, prediction)
        elif llm is None:
            prediction = heuristic_predict(case, system=f"rcaeval_{system}")
        elif system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS:
            prediction = run_rcaeval_case(case, system=system, llm=llm)
        else:
            prediction = _llm_predict_rcaeval(case, system=f"rcaeval_{system}", llm=llm)
        _write_prediction_checkpoint(args, checkpoint_suite, system, case.case_id, prediction)
        return prediction

    predictions: list[ExternalPrediction] = []
    concurrency = max(1, int(args.case_concurrency))
    for system in systems:
        if concurrency == 1:
            system_predictions = []
            for index, case in enumerate(cases, 1):
                prediction = infer(system, case)
                system_predictions.append(prediction)
                print(f"[{system}] {index}/{len(cases)} {case.runtime_case_id()} root={prediction.root}", flush=True)
        else:
            by_case: dict[str, ExternalPrediction] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(infer, system, case): case for case in cases}
                for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    case = futures[future]
                    prediction = future.result()
                    by_case[case.case_id] = prediction
                    print(f"[{system}] {completed}/{len(cases)} {case.runtime_case_id()} root={prediction.root}", flush=True)
            system_predictions = [by_case[case.case_id] for case in cases]
        predictions.extend(system_predictions)

    scored = score_predictions(cases, predictions)
    suite_name = "rcaeval_hard" if hard else "rcaeval"
    out = args.out or _default_output_path(args, [suite_name], systems)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": suite_name,
            "systems": systems,
            "sample": args.sample or len(cases),
            "seed": args.seed,
            "validation": validation,
            "hard_split": hard,
            "source_filter": sorted(source_filter),
            "excluded_case_ids": sorted(excluded_case_ids),
            "case_concurrency": concurrency,
            "mode": (
                "label-safe RCAEval profile smoke benchmark"
                if args.external_mode == "profile"
                else "label-safe RCAEval live LLM benchmark"
            ),
        },
        **scored,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved RCAEval results -> {out}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _load_rcaeval_cases(
    args: argparse.Namespace,
    *,
    hard: bool,
    source_filter: set[str],
    excluded_case_ids: set[str],
) -> list[ExternalBenchmarkCase]:
    if source_filter:
        if hard:
            cases = load_hard_cases(args.rcaeval_root, sample=None, seed=args.seed)
            cases = [case for case in cases if case.source in source_filter and case.case_id not in excluded_case_ids]
            if args.sample and args.sample < len(cases):
                import random

                cases = sorted(random.Random(args.seed).sample(cases, args.sample), key=lambda case: case.case_id)
            return cases
        return load_cases(
            args.rcaeval_root,
            sample=args.sample or None,
            seed=args.seed,
            sources=source_filter,
            exclude_case_ids=excluded_case_ids,
        )
    if hard:
        return load_hard_cases(args.rcaeval_root, sample=args.sample or None, seed=args.seed)
    return load_cases(
        args.rcaeval_root,
        sample=args.sample or None,
        seed=args.seed,
        exclude_case_ids=excluded_case_ids,
    )


def _run_openrca(args: argparse.Namespace, *, llm_mode: bool) -> int:
    from ..openrca.cli import main as openrca_main

    systems = [system.strip() for system in args.systems.split(",") if system.strip()]
    argv = [
        "--mode",
        "llm" if llm_mode else "smoke",
        "--system",
        systems[0] if systems else "shardrca_full",
        "--limit",
        str(args.sample or 3),
        "--out",
        args.out or _default_output_path(args, ["openrca_llm" if llm_mode else "openrca"], systems),
    ]
    if llm_mode:
        argv.append("--confirm-live-llm")
    return openrca_main(argv)


def _llm_predict_rcaeval(case: ExternalBenchmarkCase, *, system: str, llm: LLMClient) -> ExternalPrediction:
    system_prompt = """You are evaluating a root-cause analysis case from RCAEval.
Use only the label-safe telemetry summary. Do not assume access to hidden labels.
Return ONLY JSON:
{"root": "service_name", "ranked_roots": ["service1", "service2", "service3"],
 "fault_type": "cpu|mem|disk|delay|loss|socket|unknown",
 "confidence": 0.0, "notes": "short evidence summary"}"""
    started = time.time()
    response = llm.chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(case.inference_payload(), indent=2)}],
        force_json=True,
    )
    data = extract_json(response.content)
    ranked = data.get("ranked_roots") if isinstance(data.get("ranked_roots"), list) else []
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
        total_tokens=response.usage.total_tokens,
        tool_calls=response.usage.tool_calls,
        llm_calls=response.usage.llm_calls,
        notes=str(data.get("notes") or "live LLM prediction over label-safe RCAEval summary"),
    )


def _default_output_path(args: argparse.Namespace, suites: list[str], systems: list[str]) -> str:
    suite_label = "_".join(suites)
    if suite_label in {"rcaeval", "rcaeval_hard"}:
        suffix = "llm" if getattr(args, "external_mode", "profile") == "llm" else "profile"
        sample = f"_sample{args.sample}" if args.sample else ""
        systems_label = "_".join(systems) if systems else "default"
        return f"results/{suite_label}_{suffix}_{systems_label}{sample}.json"
    if suite_label in {"openrca", "openrca_llm"}:
        sample = f"_sample{args.sample}" if args.sample else ""
        systems_label = "_".join(systems) if systems else "shardrca_full"
        return f"results/{suite_label}_{systems_label}{sample}.json"
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
    safe_case = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_id)
    safe_system = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in system)
    return os.path.join(root, safe_system, f"{safe_case}.json")


def _load_prediction_checkpoint(args: argparse.Namespace, suite: str, system: str, case_id: str) -> ExternalPrediction | None:
    path = _checkpoint_path(args, suite, system, case_id)
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return ExternalPrediction(**json.load(handle))


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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(prediction.__dict__, handle, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

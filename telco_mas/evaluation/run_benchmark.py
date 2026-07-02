"""Benchmark the multi-agent system against the single-agent baseline.

Usage:
    python -m telco_mas.evaluation.run_benchmark                 # all scenarios, both systems
    python -m telco_mas.evaluation.run_benchmark --scenarios fiber_cut,dns_failure
    python -m telco_mas.evaluation.run_benchmark --systems multi --no-cache
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
from .plots import make_charts

SYSTEM_MODES = {"multi": "multi", "single": "single"}


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
    parser.add_argument("--scenarios", default="all", help="comma list of scenario ids, or 'all'")
    parser.add_argument("--systems", default="multi,single", help="comma list: multi,single")
    parser.add_argument("--no-cache", action="store_true", help="disable the LLM response cache")
    parser.add_argument("--out", default="results/benchmark.json")
    parser.add_argument("--figures", default="report/figures")
    args = parser.parse_args(argv)

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
    modes = [SYSTEM_MODES[s.strip()] for s in args.systems.split(",") if s.strip() in SYSTEM_MODES]
    llm = LLMClient(cache_enabled=not args.no_cache)

    rows: list[dict] = []
    for sid in scenario_ids:
        scenario = get_scenario(sid)
        for mode in modes:
            label = "multi_agent" if mode == "multi" else "single_agent"
            print(f"  running [{label}] on scenario '{sid}' …", flush=True)
            try:
                result = run(scenario, mode=mode, llm=llm)
            except LLMError as exc:
                print(f"    LLM error: {exc}", file=sys.stderr)
                return 1
            score = score_result(result, scenario)
            rows.append(score)
            mark = "OK " if score["diagnosis_correct"] else "MISS"
            print(f"    -> [{mark}] pred={score['predicted_element']} true={score['true_element']} "
                  f"resolved={score['resolved']} tokens={score['total_tokens']}")

    summary = aggregate(rows)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model": settings.model,
            "provider": settings.provider_label,
            "scenarios": scenario_ids,
            "systems": modes,
        },
        "summary": summary,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved results -> {args.out}")

    if len(summary) >= 1:
        charts = make_charts(summary, outdir=args.figures)
        print("Saved charts -> " + ", ".join(charts))

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

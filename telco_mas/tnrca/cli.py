"""Paired clean-input runner for TN-RCA-style telecom alarm graphs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from .dataset import TNRCADataset, TNRCADatasetError
from .evaluator import evaluate_predictions
from .runner import run_multi_agent, run_single_agent


SYSTEMS = {"single": run_single_agent, "multi": run_multi_agent}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired telecom alarm-graph RCA")
    parser.add_argument("--root", required=True)
    parser.add_argument("--systems", default="single,multi")
    parser.add_argument("--limit", type=int, default=0, help="0 runs all available cases")
    parser.add_argument("--out", default="results/tnrca_paired_clean.json")
    parser.add_argument("--confirm-live-llm", action="store_true")
    args = parser.parse_args(argv)

    requested = [value.strip() for value in args.systems.split(",") if value.strip()]
    invalid = sorted(set(requested) - set(SYSTEMS))
    if invalid:
        raise SystemExit(f"Unknown systems: {', '.join(invalid)}")
    if not args.confirm_live_llm:
        raise SystemExit("--confirm-live-llm is required because this command spends API tokens")
    try:
        dataset = TNRCADataset(args.root)
    except TNRCADatasetError as exc:
        raise SystemExit(str(exc)) from exc
    cases = list(dataset.cases[: args.limit or None])
    llm = LLMClient(cache_enabled=False)
    if not llm.settings.has_api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, list[str]]] = {system: {} for system in requested}
    usage: dict[str, dict[str, int]] = {
        system: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        for system in requested
    }
    for case_index, case in enumerate(cases):
        order = requested[case_index % len(requested):] + requested[: case_index % len(requested)]
        for system in order:
            result = SYSTEMS[system](case, llm)
            predictions[system][case.case_id] = list(result.root_causes)
            for key in usage[system]:
                usage[system][key] += int(getattr(result.usage, key))
            rows.append(
                {
                    "case_id": case.case_id,
                    "system": system,
                    "root_causes": list(result.root_causes),
                    "input_sha256": case.input_sha256,
                    "raw_leakage_removed": [finding.to_dict() for finding in case.leakage],
                    "usage": {key: int(getattr(result.usage, key)) for key in usage[system]},
                    "artifacts": result.artifacts,
                }
            )

    evaluation = {
        system: evaluate_predictions(cases, predictions[system])
        for system in requested
    }
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": str(dataset.root),
            "case_count": len(cases),
            "systems": requested,
            "model": llm.settings.model,
            "protocol": "clean-input; explicit answer-marker fields removed",
            "sota_comparable": False if any(case.leakage for case in cases) else len(cases) >= 100,
            "confirmatory": len(cases) >= 100,
        },
        "readiness": dataset.readiness(),
        "evaluation": evaluation,
        "usage": usage,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"meta": payload["meta"], "evaluation": evaluation, "usage": usage}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

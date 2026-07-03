"""Run OpenRCA integration in a staged, resource-aware way."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from ..llm import LLMClient, LLMError, extract_json
from .dataset import OpenRCADataset, OpenRCADatasetError
from .evaluator import build_eval_result, summarize_results
from .formatter import format_prediction
from .schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenRCA benchmark integration")
    parser.add_argument("--data-dir", default=os.getenv("OPENRCA_DATA_DIR") or "data/openrca")
    parser.add_argument("--dataset", default="Telecom")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", default="results/openrca_benchmark.json")
    parser.add_argument("--strict-data", action="store_true", help="fail if OpenRCA data is missing")
    parser.add_argument("--mode", choices=["smoke", "llm"], default="smoke",
                        help="smoke uses a deterministic format baseline; llm makes live label-safe model predictions")
    args = parser.parse_args(argv)

    try:
        dataset = OpenRCADataset(args.data_dir, dataset=args.dataset)
    except OpenRCADatasetError as exc:
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "openrca",
                "status": "skipped",
                "reason": str(exc),
                "data_dir": str(Path(args.data_dir).expanduser()),
            },
            "summary": {},
            "rows": [],
        }
        _write_json(args.out, payload)
        print(json.dumps(payload["meta"], indent=2))
        return 2 if args.strict_data else 0

    evals = []
    rows = []
    llm = None
    if args.mode == "llm":
        llm = LLMClient()
        if not llm.settings.has_api_key:
            print("ERROR: --mode llm requires OPENAI_API_KEY.", file=sys.stderr)
            return 2
    for task in dataset.iter_runtime_tasks(start_row=args.start_row, limit=args.limit):
        if llm is None:
            prediction = _baseline_prediction_from_instruction(task["instruction"])
            latency_s = 0.0
            usage = {"total_tokens": 0, "llm_calls": 0, "tool_calls": 0}
        else:
            try:
                prediction, latency_s, usage = _llm_prediction_from_instruction(task["instruction"], llm)
            except LLMError as exc:
                print(f"LLM error: {exc}", file=sys.stderr)
                return 1
        prediction_text = format_prediction(prediction)
        evaluation = build_eval_result(
            row_id=task["row_id"],
            task_index=task["task_index"],
            instruction=task["instruction"],
            prediction=prediction_text,
            scoring_points=dataset.get_scoring_points(task["row_id"]),
        )
        evals.append(evaluation)
        row = evaluation.to_dict()
        row.update({"latency_s": latency_s, **usage})
        rows.append(row)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "openrca",
            "status": "completed",
            "dataset": args.dataset,
            "rows": len(rows),
            "mode": "format/evaluator smoke baseline" if args.mode == "smoke" else "label-safe live LLM baseline",
            "evidence_warning": (
                "llm mode predicts from the query/instruction only in the staged runner; "
                "full OpenRCA telemetry-agent evaluation requires the telemetry tool path."
            ) if args.mode == "llm" else "smoke mode is only an evaluator/format sanity check",
        },
        "summary": summarize_results(evals),
        "rows": rows,
    }
    _write_json(args.out, payload)
    print(json.dumps(payload["summary"][-1] if payload["summary"] else {}, indent=2))
    return 0


def _baseline_prediction_from_instruction(instruction: str) -> OpenRCAPredictionOutput:
    # This is intentionally a smoke baseline. The research LLM runner should
    # replace it when OpenRCA data/API budget is available.
    item = OpenRCAPredictionItem()
    if "component" in instruction.lower():
        item.root_cause_component = "docker_001"
    if "reason" in instruction.lower():
        item.root_cause_reason = "CPU fault"
    if "time" in instruction.lower() or "when" in instruction.lower():
        item.root_cause_occurrence_datetime = "1970-01-01 00:00:00"
    if not any(item.model_dump().values()):
        item.root_cause_component = "docker_001"
    return OpenRCAPredictionOutput(root_causes=[item], rationale="smoke baseline")


def _llm_prediction_from_instruction(
    instruction: str,
    llm: LLMClient,
) -> tuple[OpenRCAPredictionOutput, float, dict[str, int]]:
    prompt = """You are solving an OpenRCA task from a label-safe query.
Return ONLY JSON with this shape:
{"root_causes": [{"root_cause_occurrence_datetime": "YYYY-MM-DD HH:MM:SS",
                  "root_cause_component": "component_id",
                  "root_cause_reason": "short reason"}],
 "rationale": "short evidence summary"}
Only include fields requested by the instruction; use null for unknown fields."""
    started = time.time()
    resp = llm.chat(
        [{"role": "system", "content": prompt}, {"role": "user", "content": instruction}],
        force_json=True,
    )
    data = extract_json(resp.content)
    try:
        prediction = OpenRCAPredictionOutput.model_validate(data)
    except Exception:
        prediction = OpenRCAPredictionOutput(
            root_causes=[OpenRCAPredictionItem(root_cause_component="UNKNOWN")],
            rationale="LLM response could not be parsed into OpenRCA format",
        )
    return prediction, round(time.time() - started, 2), {
        "total_tokens": resp.usage.total_tokens,
        "llm_calls": resp.usage.llm_calls,
        "tool_calls": resp.usage.tool_calls,
    }


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

"""Faithful Controller/Executor reimplementation of OpenRCA's RCA-Agent."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from ..shardrca.board import Blackboard, CandidateRootCause
from ..shardrca.runner import ShardRCARunResult
from ..shardrca.synthesizer import SynthesizerResult
from .prepared import PreparedOpenRCA
from .sandbox import DockerPythonSandbox
from .task_parser import parse_runtime_task
from .tools import TELECOM_COMPONENTS, TELECOM_REASONS

# Adapted from the MIT-licensed microsoft/OpenRCA Controller/Executor baseline.
CONTROLLER_SYSTEM = """You are the Controller of an OpenRCA failure-diagnosis system.
Iteratively give one atomic natural-language instruction to a Python Executor, inspect its result,
and follow: preprocess -> anomaly detection -> fault identification -> root-cause localization.
Analyze metrics before traces. Global KPI thresholds are available in
/telemetry/metric_stats.parquet; the target 30-minute raw slices are under /telemetry/row/.
Never request record.csv, scoring_points, labels, shell commands, network access, or file writes.
Return ONLY JSON:
{"analysis": "what the previous result establishes, or None", "completed": false,
 "instruction": "one atomic instruction for the Executor, or a concise final conclusion"}"""

EXECUTOR_SYSTEM = """You are the Python Executor in OpenRCA RCA-Agent.
Write Python/pandas code for exactly one Controller instruction. The state is persistent.
Read-only prepared telemetry exists under /telemetry/row and daily metric thresholds are in
/telemetry/metric_stats.parquet. Metric files are metric_node.parquet,
metric_container.parquet, metric_service.parquet, metric_middleware.parquet, and
metric_app.parquet under /telemetry/row/metric/. Trace spans are in
/telemetry/row/trace/trace_span.parquet. Use UTC+8. Reuse variables where useful.
Do not use shell, network, subprocess, filesystem writes, visualization, record.csv, or scoring labels.
Return ONLY JSON: {"code": "valid Python source"}"""

EXECUTOR_SUMMARY_SYSTEM = """Summarize the supplied Python execution output for the RCA Controller.
Do not invent evidence and do not mention candidate labels that are absent from the output.
Return ONLY JSON: {"summary": "concise factual observation"}"""

FINAL_SYSTEM = """Produce the final OpenRCA diagnosis from the Controller trajectory.
Select only from the supplied official component and reason catalogs.
Include only requested fields. Do not return unknown/null when a field is requested.
Return ONLY JSON:
{"component": "optional official component", "reason": "optional official reason",
 "occurrence_time": "optional YYYY-MM-DD HH:MM:SS", "confidence": 0.0,
 "rationale": "short trajectory-grounded explanation"}"""


@dataclass
class ReplicaResult:
    run_result: ShardRCARunResult
    trajectory: list[dict[str, Any]]


def run_rca_agent_replica(
    prepared: PreparedOpenRCA,
    task: dict[str, Any],
    *,
    llm: LLMClient,
    max_steps: int = 25,
    max_executor_attempts: int = 2,
    case_timeout_s: int = 600,
    sandbox_factory: Callable[[PreparedOpenRCA, int], Any] = DockerPythonSandbox,
) -> ReplicaResult:
    parsed = parse_runtime_task(task)
    started = time.monotonic()
    usage = UsageStats()
    trajectory: list[dict[str, Any]] = []
    controller_messages = [
        {"role": "system", "content": CONTROLLER_SYSTEM},
        {
            "role": "user",
            "content": json.dumps({
                "issue": parsed.instruction,
                "requested_fields": list(parsed.requested_fields),
                "time_range": [
                    parsed.start.strftime("%Y-%m-%d %H:%M:%S"),
                    parsed.end.strftime("%Y-%m-%d %H:%M:%S"),
                ],
                "candidate_components": TELECOM_COMPONENTS,
                "candidate_reasons": TELECOM_REASONS,
            }),
        },
    ]
    executor_history: list[dict[str, str]] = [{"role": "system", "content": EXECUTOR_SYSTEM}]
    with sandbox_factory(prepared, parsed.row_id) as sandbox:
        for step in range(max_steps):
            if time.monotonic() - started >= case_timeout_s:
                trajectory.append({"step": step + 1, "status": "case_timeout"})
                break
            controller_context = controller_messages[:2] + controller_messages[-20:]
            response = llm.chat(controller_context, force_json=True)
            usage = usage.add(response.usage)
            decision = extract_json(response.content)
            instruction = str(decision.get("instruction") or "").strip()
            completed = _as_bool(decision.get("completed"))
            if completed:
                trajectory.append({
                    "step": step + 1,
                    "controller": decision,
                    "status": "completed",
                })
                break
            if not instruction:
                instruction = "Inspect the most relevant metric shard for anomalies in the target window."
            execution = None
            code = ""
            for attempt in range(max_executor_attempts):
                executor_prompt = executor_history[:1] + executor_history[-12:] + [
                    {"role": "user", "content": instruction}
                ]
                if execution is not None:
                    executor_prompt.append({
                        "role": "user",
                        "content": f"The prior code failed. Correct it using this error:\n{execution['output']}",
                    })
                generated = llm.chat(executor_prompt, force_json=True)
                usage = usage.add(generated.usage)
                code = _extract_code(extract_json(generated.content), generated.content)
                execution = sandbox.execute(code)
                usage.tool_calls += 1
                if execution.get("ok"):
                    break
            execution = execution or {"ok": False, "output": "Executor produced no result."}
            summary_response = llm.chat(
                [
                    {"role": "system", "content": EXECUTOR_SUMMARY_SYSTEM},
                    {"role": "user", "content": str(execution.get("output") or "")[:64_000]},
                ],
                force_json=True,
            )
            usage = usage.add(summary_response.usage)
            summary = str(extract_json(summary_response.content).get("summary") or execution.get("output") or "")
            trajectory.append({
                "step": step + 1,
                "controller": decision,
                "instruction": instruction,
                "code": code,
                "execution_ok": bool(execution.get("ok")),
                "execution_output": str(execution.get("output") or "")[:64_000],
                "summary": summary,
            })
            controller_messages.extend([
                {"role": "assistant", "content": json.dumps(decision)},
                {"role": "user", "content": summary},
            ])
            executor_history.extend([
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": json.dumps({"code": code})},
                {"role": "user", "content": f"Execution result:\n{str(execution.get('output') or '')[:16_000]}"},
            ])

    final_response = llm.chat(
        [
            {"role": "system", "content": FINAL_SYSTEM},
            {"role": "user", "content": json.dumps({
                "issue": parsed.instruction,
                "requested_fields": list(parsed.requested_fields),
                "candidate_components": TELECOM_COMPONENTS,
                "candidate_reasons": TELECOM_REASONS,
                "trajectory": [
                    {
                        "step": item.get("step"),
                        "analysis": (item.get("controller") or {}).get("analysis"),
                        "instruction": item.get("instruction"),
                        "execution_ok": item.get("execution_ok"),
                        "summary": item.get("summary"),
                        "status": item.get("status"),
                    }
                    for item in trajectory
                ],
            }, ensure_ascii=True, default=str)},
        ],
        force_json=True,
    )
    usage = usage.add(final_response.usage)
    final = extract_json(final_response.content)
    candidate = _candidate_from_final(final, parsed)
    board = Blackboard(
        case_id=str(parsed.row_id),
        catalog_summary={
            "dataset": "OpenRCA Telecom",
            "architecture": "official Controller+Executor RCA-Agent replica",
        },
    )
    synthesis = SynthesizerResult(
        winner=candidate,
        candidates=[candidate],
        vote_breakdown={candidate.component: candidate.confidence},
    )
    run_result = ShardRCARunResult(
        system="rca_agent_replica",
        board=board,
        synthesis=synthesis,
        winner=candidate,
        usage=usage,
        latency_s=round(time.monotonic() - started, 2),
        notes="official-style OpenRCA Controller+Executor with stateful sandboxed Python",
        artifacts={
            "architecture": "controller_executor",
            "max_steps": max_steps,
            "max_executor_attempts": max_executor_attempts,
            "trajectory": trajectory,
        },
    )
    return ReplicaResult(run_result=run_result, trajectory=trajectory)


def _candidate_from_final(data: dict[str, Any], parsed) -> CandidateRootCause:
    component = str(data.get("component") or "").strip()
    reason = str(data.get("reason") or "").strip()
    occurrence = str(data.get("occurrence_time") or "").strip() or None
    if component not in TELECOM_COMPONENTS:
        component = TELECOM_COMPONENTS[0]
    if reason not in TELECOM_REASONS:
        reason = TELECOM_REASONS[0]
    if occurrence and not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", occurrence):
        occurrence = parsed.start.strftime("%Y-%m-%d %H:%M:%S")
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    return CandidateRootCause(
        component=component,
        reason=reason,
        occurrence_time=occurrence,
        confidence=max(0.0, min(1.0, confidence)),
        rationale=str(data.get("rationale") or "RCA-Agent Controller/Executor conclusion"),
    )


def _extract_code(data: dict[str, Any], raw: str | None) -> str:
    code = str(data.get("code") or "").strip()
    if code:
        return code
    match = re.search(r"```python\s*(.*?)```", raw or "", re.DOTALL)
    return match.group(1).strip() if match else str(raw or "").strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"

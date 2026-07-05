"""Run TeleLogsAgent fallback evaluations.

`profile` mode is an ingestion/scoring smoke test. `llm` mode makes label-safe
predictions from the stripped runtime task payload. `tool` mode uses the
official TeleLogsAgent FastAPI server endpoints through OpenAI-style tool calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..evaluation.stats import aggregate_ci
from ..llm import LLMClient, LLMError, extract_json
from ..schemas import UsageStats
from .dataset import TeleLogsAgentDataset, TeleLogsAgentDatasetError
from .evaluator import score_prediction
from .http_tools import TeleLogsHTTPClient, TeleLogsHTTPError
from .prereg import _manifest


DEFAULT_SYSTEMS = ("single_react", "single_react_sc", "same_board_single", "shardrca_full")
_CONFIG_TOOLS = {"scenario", "cell_info", "gnodeb_location", "user_location", "user_speed"}
_RF_TOOLS = {
    "throughput_logs",
    "serving_cell_pci",
    "serving_cell_rsrp",
    "serving_cell_sinr",
    "rbs_allocated_to_user",
    "neighboring_cells_pci",
    "neighboring_cell_rsrp",
    "beam_scenario_info",
}
_SIGNALING_TOOLS = {"signaling_plane_event_log"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TeleLogsAgent profile, staged LLM, or HTTP-tool evaluations.")
    parser.add_argument("--data-dir", default=os.getenv("TELELOGS_AGENT_DATA_DIR") or "data/telelogs_agent")
    parser.add_argument("--prereg", default=None, help="optional frozen preregistration; fixes rows/systems")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--limit-per-set", type=int, default=1, help="used only without --prereg; 0 means all")
    parser.add_argument("--mode", choices=["profile", "llm", "tool"], default="profile")
    parser.add_argument("--server-url", default=os.getenv("TELELOGS_AGENT_SERVER_URL") or "http://localhost:7861")
    parser.add_argument(
        "--server-url-map",
        default=os.getenv("TELELOGS_AGENT_SERVER_URL_MAP") or "",
        help="optional comma-separated mapping, e.g. TS1=http://localhost:7861,TS2=http://localhost:7862",
    )
    parser.add_argument("--http-timeout", type=float, default=20.0)
    parser.add_argument("--max-tool-iters", type=int, default=None)
    parser.add_argument("--max-tool-result-chars", type=int, default=16_000)
    parser.add_argument("--out", default="results/telelogs_agent_profile.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        server_url_by_set = _parse_server_url_map(args.server_url_map)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    args.server_url_by_set = server_url_by_set

    try:
        dataset = TeleLogsAgentDataset(args.data_dir)
        prereg = _load_prereg(args.prereg) if args.prereg else None
        if prereg:
            _verify_prereg(dataset, prereg)
            selected = prereg["row_selection"]["selected"]
            systems = list(prereg["systems"])
        else:
            selected = _first_rows(dataset, limit_per_set=args.limit_per_set or None)
            systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    except (TeleLogsAgentDatasetError, ValueError, OSError, json.JSONDecodeError) as exc:
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "telelogs_agent",
                "status": "skipped",
                "reason": str(exc),
                "data_dir": str(Path(args.data_dir).expanduser()),
            },
            "summary": {},
            "rows": [],
        }
        _write_json(args.out, payload)
        print(json.dumps(payload["meta"], indent=2))
        return 2

    if not systems:
        print("ERROR: no systems selected", file=sys.stderr)
        return 2

    llm = None
    if args.mode in {"llm", "tool"}:
        settings = get_settings()
        if not settings.has_api_key and not args.cache_only:
            print(f"ERROR: --mode {args.mode} requires OPENAI_API_KEY unless --cache-only is used.", file=sys.stderr)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)
    if args.mode == "tool":
        skipped = _preflight_tool_server(dataset, selected, args)
        if skipped is not None:
            _write_json(args.out, skipped)
            print(json.dumps(skipped["meta"], indent=2))
            return 2

    rows: list[dict[str, Any]] = []
    for system in systems:
        for task in dataset.iter_runtime_tasks(selected):
            server_url = _server_url_for_task(task, args)
            row = _run_one(
                dataset,
                task,
                system=system,
                mode=args.mode,
                llm=llm,
                server_url=server_url,
                http_timeout=args.http_timeout,
                max_tool_iters=args.max_tool_iters,
                max_tool_result_chars=args.max_tool_result_chars,
            )
            rows.append(row)
            print(
                f"[{system}] {task['scenario_set']}#{task['row_id']} "
                f"score={row['score']} strict={row['strict_correct']} "
                f"available={row['score_available']}"
            )

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "telelogs_agent",
            "status": "completed",
            "mode": args.mode,
            "systems": systems,
            "selected": selected,
            "prereg": args.prereg,
            "data_dir": str(dataset.root_dir),
            "dataset_manifest_sha256": _manifest(dataset)["sha256"],
            "server_url": args.server_url if args.mode == "tool" else None,
            "server_url_by_set": _server_url_assignment(selected, args) if args.mode == "tool" else None,
            "evidence_warning": _evidence_warning(args.mode),
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    _write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _run_one(
    dataset: TeleLogsAgentDataset,
    task: dict[str, Any],
    *,
    system: str,
    mode: str,
    llm: LLMClient | None,
    server_url: str = "http://localhost:7861",
    http_timeout: float = 20.0,
    max_tool_iters: int | None = None,
    max_tool_result_chars: int = 16_000,
) -> dict[str, Any]:
    started = time.time()
    usage = {"total_tokens": 0, "llm_calls": 0, "tool_calls": 0}
    if mode == "profile":
        prediction = _profile_prediction(task, system)
    elif mode == "llm":
        assert llm is not None
        prediction, usage = _llm_prediction(task, system, llm)
    else:
        assert llm is not None
        prediction, usage = _tool_prediction(
            task,
            system,
            llm,
            server_url=server_url,
            http_timeout=http_timeout,
            max_tool_iters=max_tool_iters,
            max_tool_result_chars=max_tool_result_chars,
        )
    raw = dataset.rows(task["scenario_set"])[task["row_id"]]
    score = score_prediction(raw, prediction)
    usage["tool_failures"] = int(usage.get("tool_failures") or 0)
    usage["tool_failure_rate"] = _rate(usage["tool_failures"], int(usage.get("tool_calls") or 0))
    usage["tool_call_efficiency"] = _tool_call_efficiency(score.score, int(usage.get("tool_calls") or 0))
    row = {
        "case_id": f"{task['scenario_set']}:{task['row_id']}",
        "scenario_set": task["scenario_set"],
        "row_id": task["row_id"],
        "task_id": task["task_id"],
        "system": system,
        "prediction": prediction,
        "latency_s": round(time.time() - started, 3),
        **usage,
        **score.to_dict(),
    }
    return row


def _profile_prediction(task: dict[str, Any], system: str) -> dict[str, Any]:
    payload = task.get("payload", {})
    text = json.dumps(payload, sort_keys=True)
    return {
        "root_cause": "UNKNOWN",
        "final_answer": "profile-mode smoke prediction; no label-safe root inferred",
        "rationale": f"Loaded {task['scenario_set']} task {task['task_id']} for {system}; payload bytes={len(text)}.",
    }


def _llm_prediction(task: dict[str, Any], system: str, llm: LLMClient) -> tuple[dict[str, Any], dict[str, int]]:
    system_prompt = (
        "You are evaluating a 5G troubleshooting task from TeleLogsAgent. "
        "Use only the provided label-safe payload. Return ONLY JSON with keys "
        "`root_cause`, `solution`, `confidence`, and `rationale`."
    )
    if system.startswith("shardrca"):
        system_prompt += " Decompose evidence by configuration, KPI time series, and signaling/user-plane observations before deciding."
    elif system.startswith("single"):
        system_prompt += " Act as one careful NOC engineer."
    user = json.dumps(task, indent=2, sort_keys=True)
    try:
        response = llm.chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            force_json=True,
        )
    except LLMError:
        raise
    data = extract_json(response.content)
    if not data:
        data = {"root_cause": "UNKNOWN", "rationale": "LLM response was not valid JSON"}
    return data, {
        "total_tokens": response.usage.total_tokens,
        "llm_calls": response.usage.llm_calls,
        "tool_calls": response.usage.tool_calls,
        "tool_failures": 0,
        "tool_failure_rate": 0.0,
        "tool_call_efficiency": 0.0,
    }


def _tool_prediction(
    task: dict[str, Any],
    system: str,
    llm: LLMClient,
    *,
    server_url: str,
    http_timeout: float,
    max_tool_iters: int | None,
    max_tool_result_chars: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    client = TeleLogsHTTPClient(
        server_url,
        scenario_id=str(task["task_id"]),
        timeout=http_timeout,
        max_result_chars=max_tool_result_chars,
    )
    tools_spec = client.tools_spec()
    if system.startswith("shardrca"):
        return _tool_mas_prediction(task, system, llm, client, tools_spec, max_tool_iters=max_tool_iters)
    return _tool_single_prediction(task, system, llm, client, tools_spec, max_tool_iters=max_tool_iters)


def _tool_single_prediction(
    task: dict[str, Any],
    system: str,
    llm: LLMClient,
    client: TeleLogsHTTPClient,
    tools_spec: list[dict[str, Any]],
    *,
    max_tool_iters: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    system_prompt = (
        "You are a single 5G NOC engineer solving a TeleLogsAgent task. "
        "Use the official HTTP tools to inspect scenario, RF/KPI, mobility, "
        "signaling, and beam evidence before answering. Return ONLY JSON with "
        "keys `root_cause`, `solution`, `confidence`, `evidence`, and `rationale`."
    )
    if system.endswith("_sc"):
        system_prompt += " Check at least two plausible explanations before selecting the final root cause."
    user_prompt = _tool_user_prompt(task, system)
    run = llm.run_agent(
        name=system,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools_spec=tools_spec,
        dispatcher=client.dispatch,
        max_iters=max_tool_iters,
    )
    stats = _trace_tool_stats(run.trace)
    return _prediction_data(run.data), _usage_dict(run.usage, tool_failures=stats["tool_failures"])


def _tool_mas_prediction(
    task: dict[str, Any],
    system: str,
    llm: LLMClient,
    client: TeleLogsHTTPClient,
    tools_spec: list[dict[str, Any]],
    *,
    max_tool_iters: int | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    roles = [
        (
            "config_context",
            "Inspect scenario metadata, cell configuration, locations, and mobility context. "
            "Return ONLY JSON with root_cause, solution, confidence, evidence, and rationale.",
            _CONFIG_TOOLS,
        ),
        (
            "rf_kpi",
            "Inspect RF, throughput, resource-block, beam, and neighbor-cell evidence. "
            "Return ONLY JSON with root_cause, solution, confidence, evidence, and rationale.",
            _RF_TOOLS,
        ),
        (
            "signaling",
            "Inspect signaling-plane evidence when the scenario exposes it, and explain any failed handover/session pattern. "
            "Return ONLY JSON with root_cause, solution, confidence, evidence, and rationale.",
            _SIGNALING_TOOLS,
        ),
    ]
    usage = UsageStats()
    tool_failures = 0
    role_outputs: list[dict[str, Any]] = []
    for role_name, prompt, allowed in roles:
        role_tools = _filter_tools(tools_spec, allowed) or tools_spec
        run = llm.run_agent(
            name=f"{system}_{role_name}",
            system_prompt=f"You are the {role_name} specialist in a multi-agent 5G RCA team. {prompt}",
            user_prompt=_tool_user_prompt(task, system, role=role_name),
            tools_spec=role_tools,
            dispatcher=client.dispatch,
            max_iters=max_tool_iters,
        )
        usage = usage.add(run.usage)
        stats = _trace_tool_stats(run.trace)
        tool_failures += stats["tool_failures"]
        role_outputs.append({
            "role": role_name,
            "finding": _prediction_data(run.data),
            "tool_calls": run.usage.tool_calls,
            "tool_failures": stats["tool_failures"],
        })

    synth_prompt = (
        "You are the synthesis agent for a multi-agent TeleLogsAgent RCA team. "
        "Reconcile the specialist findings, resolve conflicts using cited evidence, "
        "and return ONLY JSON with keys `root_cause`, `solution`, `confidence`, "
        "`evidence`, `rationale`, and `role_agreement`."
    )
    synth_payload = {
        "benchmark": "TeleLogsAgent",
        "system": system,
        "task": task,
        "specialist_findings": role_outputs,
        "instruction": "Select the final root cause/remediation from the independent specialist findings.",
    }
    synth_run = llm.run_agent(
        name=f"{system}_synthesis",
        system_prompt=synth_prompt,
        user_prompt=json.dumps(synth_payload, indent=2, sort_keys=True),
        tools_spec=None,
        dispatcher=None,
        max_iters=1,
    )
    usage = usage.add(synth_run.usage)
    data = _prediction_data(synth_run.data)
    if not data.get("root_cause"):
        data = _best_role_output(role_outputs)
    data["mas_role_outputs"] = role_outputs
    return data, _usage_dict(usage, tool_failures=tool_failures)


def _tool_user_prompt(task: dict[str, Any], system: str, *, role: str | None = None) -> str:
    payload = {
        "benchmark": "TeleLogsAgent",
        "system": system,
        "role": role,
        "scenario_set": task["scenario_set"],
        "scenario_id_header": task["task_id"],
        "runtime_task": task,
        "label_safety": "Runtime payload has evaluator labels stripped; use only task text and tool-returned evidence.",
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    scored = [
        {
            "system": row["system"],
            "strict_correct": bool(row.get("strict_correct")),
            "score": float(row.get("score") or 0.0),
            "score_available": bool(row.get("score_available")),
            "tool_calls": int(row.get("tool_calls") or 0),
            "tool_failures": int(row.get("tool_failures") or 0),
            "tool_failure_rate": float(row.get("tool_failure_rate") or 0.0),
            "tool_call_efficiency": float(row.get("tool_call_efficiency") or 0.0),
        }
        for row in rows
    ]
    out = aggregate_ci(
        scored,
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
    by_set: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_set[row["system"]][row["scenario_set"]].append(float(row.get("score") or 0.0))
    for system, sets in by_set.items():
        out.setdefault(system, {})
        out[system]["per_TS_score"] = {
            name: round(sum(values) / len(values), 4) if values else 0.0
            for name, values in sorted(sets.items())
        }
    return out


def _prediction_data(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {"root_cause": "UNKNOWN", "rationale": "agent did not return a JSON object"}


def _best_role_output(role_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_conf = -1.0
    for row in role_outputs:
        finding = row.get("finding") if isinstance(row.get("finding"), dict) else {}
        try:
            confidence = float(finding.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        if confidence > best_conf:
            best = dict(finding)
            best_conf = confidence
    return best or {"root_cause": "UNKNOWN", "rationale": "no specialist produced a usable finding"}


def _filter_tools(tools_spec: list[dict[str, Any]], allowed: set[str]) -> list[dict[str, Any]]:
    out = []
    for spec in tools_spec:
        fn = spec.get("function") if isinstance(spec, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if name in allowed:
            out.append(spec)
    return out


def _usage_dict(usage: UsageStats, *, tool_failures: int = 0) -> dict[str, int | float]:
    return {
        "total_tokens": usage.total_tokens,
        "llm_calls": usage.llm_calls,
        "tool_calls": usage.tool_calls,
        "tool_failures": tool_failures,
        "tool_failure_rate": _rate(tool_failures, usage.tool_calls),
    }


def _evidence_warning(mode: str) -> str:
    if mode == "profile":
        return "profile mode is only an ingestion/scoring smoke test"
    if mode == "llm":
        return (
            "llm mode uses label-safe task JSON, not the official TeleLogsAgent "
            "HTTP tool server; report as staged fallback engineering evidence"
        )
    return (
        "tool mode uses the TeleLogsAgent FastAPI HTTP tools and label-safe prompts; "
        "TeleLogsAgent is still a synthetic 5G fallback, not real-network headline evidence"
    )


def _first_rows(dataset: TeleLogsAgentDataset, *, limit_per_set: int | None) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for name in dataset.scenario_sets:
        ids = list(range(len(dataset.rows(name))))
        selected[name] = ids if limit_per_set is None else ids[: min(limit_per_set, len(ids))]
    return selected


def _parse_server_url_map(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    out: dict[str, str] = {}
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("--server-url-map entries must be NAME=URL")
        name, url = text.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url:
            raise ValueError("--server-url-map entries must include non-empty scenario set and URL")
        out[name] = url
    return out


def _server_url_for_task(task: dict[str, Any], args: argparse.Namespace) -> str:
    return args.server_url_by_set.get(str(task["scenario_set"]), args.server_url)


def _server_url_assignment(selected: dict[str, list[int]], args: argparse.Namespace) -> dict[str, str]:
    return {
        name: args.server_url_by_set.get(name, args.server_url)
        for name, ids in selected.items()
        if ids
    }


def _first_selected_tasks(
    dataset: TeleLogsAgentDataset,
    selected: dict[str, list[int]],
) -> list[dict[str, Any]]:
    tasks = []
    for name in dataset.scenario_sets:
        ids = selected.get(name) or []
        if ids:
            tasks.append(dataset.get_runtime_task(name, ids[0]))
    return tasks


def _preflight_tool_server(
    dataset: TeleLogsAgentDataset,
    selected: dict[str, list[int]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    first_tasks = _first_selected_tasks(dataset, selected)
    if not first_tasks:
        return {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "telelogs_agent",
                "status": "skipped",
                "reason": "TeleLogsAgent selection contains no rows",
                "data_dir": str(dataset.root_dir),
            },
            "summary": {},
            "rows": [],
        }
    try:
        for task in first_tasks:
            TeleLogsHTTPClient(
                _server_url_for_task(task, args),
                scenario_id=str(task["task_id"]),
                timeout=args.http_timeout,
                max_result_chars=args.max_tool_result_chars,
            ).tools_spec()
    except TeleLogsHTTPError as exc:
        return {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "telelogs_agent",
                "status": "skipped",
                "reason": str(exc),
                "data_dir": str(dataset.root_dir),
                "server_url": args.server_url,
                "server_url_by_set": _server_url_assignment(selected, args),
                "next_action": (
                    "Start the official TeleLogsAgent FastAPI server for each selected scenario set, "
                    "e.g. `TELELOGS_AGENT_CONFIG=TS1 python fastapi_server.py`, then rerun --mode tool "
                    "with --server-url for one set or --server-url-map for TS1/TS2/TS3."
                ),
            },
            "summary": {},
            "rows": [],
        }
    return None


def _trace_tool_stats(trace: list[Any]) -> dict[str, int]:
    failures = 0
    for step in trace:
        for call in getattr(step, "tool_calls", []) or []:
            preview = str(getattr(call, "result_preview", "") or "")
            if preview.startswith("ERROR"):
                failures += 1
    return {"tool_failures": failures}


def _rate(numer: int, denom: int) -> float:
    return round(float(numer) / float(denom), 4) if denom else 0.0


def _tool_call_efficiency(score: float, tool_calls: int) -> float:
    return round(float(score) / max(1, tool_calls), 4)


def _load_prereg(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_prereg(dataset: TeleLogsAgentDataset, prereg: dict[str, Any]) -> None:
    manifest = _manifest(dataset)
    if prereg.get("status") != "frozen":
        raise ValueError("TeleLogsAgent preregistration is not frozen")
    if prereg.get("dataset", {}).get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("TeleLogsAgent dataset manifest differs from preregistration")
    selected = prereg.get("row_selection", {}).get("selected")
    if not isinstance(selected, dict) or not any(selected.values()):
        raise ValueError("TeleLogsAgent preregistration has no selected rows")
    for name, row_ids in selected.items():
        if name not in dataset.scenario_sets:
            raise ValueError(f"unknown TeleLogsAgent scenario set in preregistration: {name}")
        row_count = len(dataset.rows(name))
        if any(not isinstance(row_id, int) or row_id < 0 or row_id >= row_count for row_id in row_ids):
            raise ValueError(f"invalid row id in {name} selection")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

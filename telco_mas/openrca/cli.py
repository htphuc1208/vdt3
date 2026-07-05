"""Run OpenRCA integration in a staged, resource-aware way."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm import LLMClient, LLMError, extract_json
from ..shardrca.runner import (
    OPENRCA_HEURISTIC_SYSTEMS,
    OPENRCA_AGENT_SYSTEMS,
    SHARDRCA_SYSTEMS,
    SINGLE_SYSTEMS,
    run_openrca_task,
)
from .dataset import OpenRCADataset, OpenRCADatasetError
from .evaluator import build_eval_result, summarize_results
from .formatter import format_prediction
from .prepared import PreparedOpenRCA, PreparedOpenRCAError
from .prereg import _algorithm_manifest, _sha256_file, _telemetry_manifest
from .schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenRCA benchmark integration")
    parser.add_argument("--data-dir", default=os.getenv("OPENRCA_DATA_DIR") or "data/openrca")
    parser.add_argument("--dataset", default="Telecom")
    parser.add_argument("--prereg", default=None, help="optional frozen preregistration; fixes rows/systems")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--row-ids", default=None,
                        help="comma-separated row IDs to run exactly; overrides --start-row/--limit")
    parser.add_argument("--out", default="results/openrca_benchmark.json")
    parser.add_argument("--strict-data", action="store_true", help="fail if OpenRCA data is missing")
    parser.add_argument("--mode", choices=["smoke", "llm"], default="smoke",
                        help="smoke uses a deterministic format baseline; llm makes live label-safe model predictions")
    parser.add_argument("--system", default=None,
                        help="legacy single inference system; use --systems for paired runs")
    parser.add_argument("--systems", default=None,
                        help="comma-separated inference systems for one paired result file")
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--prepared-dir", default="data/openrca_prepared/Telecom")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/openrca_telecom")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--confirm-live-llm",
        action="store_true",
        help="required with --mode llm; acknowledges live API/token spend",
    )
    args = parser.parse_args(argv)

    try:
        requested_systems = _systems_from_args(args)
        dataset = OpenRCADataset(args.data_dir, dataset=args.dataset)
        prereg = _load_prereg(args.prereg) if args.prereg else None
        if prereg:
            _verify_prereg(dataset, prereg)
            prereg_row_ids = list(prereg["row_selection"]["row_ids"])
            prereg_systems = list(prereg["systems"])
            if args.row_ids and _parse_row_ids(args.row_ids) != prereg_row_ids:
                raise ValueError("CLI --row-ids does not match frozen OpenRCA preregistration")
            if requested_systems and requested_systems != prereg_systems:
                raise ValueError("CLI --systems/--system does not match frozen OpenRCA preregistration")
            selected_row_ids = prereg_row_ids
            systems = prereg_systems
        else:
            selected_row_ids = _parse_row_ids(args.row_ids) if args.row_ids else None
            systems = requested_systems or [("instruction_llm" if args.mode == "llm" else "smoke")]
    except (OpenRCADatasetError, ValueError, OSError, json.JSONDecodeError) as exc:
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "openrca",
                "status": "skipped",
                "reason": str(exc),
                "data_dir": str(Path(args.data_dir).expanduser()),
                "prereg": args.prereg,
            },
            "summary": {},
            "rows": [],
        }
        _write_json(args.out, payload)
        print(json.dumps(payload["meta"], indent=2))
        return 2 if args.strict_data else 0

    if not systems:
        print("ERROR: no systems selected", file=sys.stderr)
        return 2
    if args.mode == "llm" and not args.confirm_live_llm:
        print(
            "ERROR: --mode llm can make live API calls and spend tokens. "
            "Re-run with --confirm-live-llm after explicitly approving the row/system budget.",
            file=sys.stderr,
        )
        return 2

    evals_by_system: dict[str, list[Any]] = defaultdict(list)
    rows = []
    llm = None
    prepared = None
    needs_prepared = any(
        system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS or system in OPENRCA_AGENT_SYSTEMS
        or system in OPENRCA_HEURISTIC_SYSTEMS
        or system in {"full", "multi"}
        for system in systems
    )
    if needs_prepared:
        try:
            prepared = PreparedOpenRCA(args.prepared_dir)
            prepared.validate_against(dataset)
            if prereg:
                _verify_prepared_prereg(prepared, prereg)
        except PreparedOpenRCAError as exc:
            print(
                "ERROR: prepared telemetry is required; run "
                "`python3 -m telco_mas.openrca.prepared` first. "
                f"Details: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.mode == "llm":
        llm = LLMClient(cache_enabled=False if args.no_cache else None)
        if not llm.settings.has_api_key:
            print("ERROR: --mode llm requires OPENAI_API_KEY.", file=sys.stderr)
            return 2
        if prereg:
            _verify_execution_settings(prereg, llm, no_cache=args.no_cache)
    tasks = (
        dataset.iter_runtime_tasks_by_id(selected_row_ids)
        if selected_row_ids is not None
        else dataset.iter_runtime_tasks(start_row=args.start_row, limit=args.limit)
    )
    selected_tasks = list(tasks)
    fingerprint = _execution_fingerprint(
        dataset=dataset,
        prepared=prepared,
        systems=systems,
        prereg_path=args.prereg,
        llm=llm,
        chunksize=args.chunksize,
    )
    actual_systems = list(systems)
    smoke_ids = set((prereg or {}).get("smoke_gate", {}).get("row_ids", []))
    ordered_tasks = sorted(selected_tasks, key=lambda task: (task["row_id"] not in smoke_ids, task["row_id"]))
    for task in ordered_tasks:
        rotation = int(task["row_id"]) % len(systems)
        ordered_systems = systems[rotation:] + systems[:rotation]
        for system in ordered_systems:
            checkpoint = _checkpoint_path(args.checkpoint_dir, system, int(task["row_id"]))
            checkpoint_row = _load_checkpoint(checkpoint, fingerprint) if args.resume else None
            loaded_checkpoint = checkpoint_row is not None
            if checkpoint_row is not None:
                row = checkpoint_row
                actual_system = str(row["system"])
                evaluation = build_eval_result(
                    row_id=int(row["row_id"]),
                    task_index=str(row["task_index"]),
                    instruction=str(row["instruction"]),
                    prediction=str(row["prediction"]),
                    scoring_points=dataset.get_scoring_points(int(row["row_id"])),
                )
            else:
                use_shardrca = (
                    system in SHARDRCA_SYSTEMS
                    or system in SINGLE_SYSTEMS
                    or system in OPENRCA_AGENT_SYSTEMS
                    or system in OPENRCA_HEURISTIC_SYSTEMS
                    or system in {"full", "multi"}
                )
                artifacts = None
                if use_shardrca:
                    prediction, run_result = run_openrca_task(
                        dataset,
                        task,
                        system=system,
                        llm=llm,
                        chunksize=args.chunksize,
                        prepared=prepared,
                    )
                    actual_system = run_result.system
                    latency_s = run_result.latency_s
                    artifacts = run_result.artifacts
                    usage = {
                        "total_tokens": run_result.usage.total_tokens,
                        "llm_calls": run_result.usage.llm_calls,
                        "tool_calls": run_result.usage.tool_calls,
                    }
                elif llm is None:
                    prediction = _baseline_prediction_from_instruction(task["instruction"])
                    actual_system = system
                    latency_s = 0.0
                    usage = {"total_tokens": 0, "llm_calls": 0, "tool_calls": 0}
                else:
                    try:
                        prediction, latency_s, usage = _llm_prediction_from_instruction(task["instruction"], llm)
                    except LLMError as exc:
                        print(f"LLM error: {exc}", file=sys.stderr)
                        return 1
                    actual_system = system
                prediction_text = format_prediction(prediction)
                evaluation = build_eval_result(
                    row_id=task["row_id"],
                    task_index=task["task_index"],
                    instruction=task["instruction"],
                    prediction=prediction_text,
                    scoring_points=dataset.get_scoring_points(task["row_id"]),
                )
                row = evaluation.to_dict()
                row.update({"system": actual_system, "latency_s": latency_s, **usage})
                if artifacts is not None:
                    row["artifacts"] = artifacts
            evals_by_system[actual_system].append(evaluation)
            if prepared is not None:
                row["volume_bin"] = _volume_bin(prepared, int(task["row_id"]))
            if not loaded_checkpoint:
                _write_checkpoint(checkpoint, fingerprint, row)
            rows.append(row)
    summary_by_system = {
        system: summarize_results(evals)
        for system, evals in evals_by_system.items()
    }
    summary = (
        next(iter(summary_by_system.values()))
        if len(summary_by_system) == 1
        else summary_by_system
    )
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "openrca",
            "status": "completed",
            "dataset": args.dataset,
            "rows": len(selected_tasks),
            "systems": actual_systems,
            "system": actual_systems[0] if len(actual_systems) == 1 else None,
            "prereg": args.prereg,
            "execution_fingerprint": fingerprint,
            "prepared_manifest": str(prepared.manifest_path) if prepared else None,
            "complete": len(rows) == len(selected_tasks) * len(systems),
            "expected_result_rows": len(selected_tasks) * len(systems),
            "mode": (
                "mixed OpenRCA paired run"
                if len(actual_systems) > 1
                else (
                    "ShardRCA telemetry-agent path"
                    if (
                        actual_systems[0] in SHARDRCA_SYSTEMS
                        or actual_systems[0] in SINGLE_SYSTEMS
                        or actual_systems[0] in OPENRCA_AGENT_SYSTEMS
                        or actual_systems[0] in OPENRCA_HEURISTIC_SYSTEMS
                    )
                    else ("format/evaluator smoke baseline" if args.mode == "smoke" else "label-safe live LLM baseline")
                )
            ),
            "evidence_warning": (
                "llm mode predicts from the query/instruction only in the staged runner; "
                "full OpenRCA telemetry-agent evaluation requires the telemetry tool path."
            ) if args.mode == "llm" and not any(system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS for system in actual_systems) else (
                "" if any(system in SHARDRCA_SYSTEMS or system in SINGLE_SYSTEMS for system in actual_systems) else "smoke mode is only an evaluator/format sanity check"
            ),
        },
        "summary": summary,
        "summary_by_system": summary_by_system,
        "rows": rows,
    }
    _write_json(args.out, payload)
    print(json.dumps(_printable_summary(summary_by_system), indent=2))
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


def _parse_row_ids(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _systems_from_args(args: argparse.Namespace) -> list[str]:
    if args.system and args.systems:
        raise ValueError("Use only one of --system or --systems")
    if args.systems:
        systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    elif args.system:
        systems = [args.system.strip()]
    else:
        systems = []
    if len(set(systems)) != len(systems):
        raise ValueError("OpenRCA systems must be unique")
    return systems


def _load_prereg(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_prereg(dataset: OpenRCADataset, prereg: dict[str, Any]) -> None:
    if prereg.get("status") != "frozen":
        raise ValueError("OpenRCA preregistration is not frozen")
    expected = prereg.get("dataset", {})
    current_query = _sha256_file(dataset.query_path)
    current_telemetry = _telemetry_manifest(dataset.telemetry_dir)["sha256"]
    if expected.get("query_sha256") != current_query:
        raise ValueError("OpenRCA query.csv differs from preregistration")
    if expected.get("telemetry_manifest_sha256") != current_telemetry:
        raise ValueError("OpenRCA telemetry manifest differs from preregistration")
    row_ids = prereg.get("row_selection", {}).get("row_ids")
    if not isinstance(row_ids, list) or not row_ids:
        raise ValueError("OpenRCA preregistration has no row_ids")
    if any(not isinstance(row_id, int) or row_id < 0 or row_id >= len(dataset.rows) for row_id in row_ids):
        raise ValueError("OpenRCA preregistration contains invalid row_ids")
    systems = prereg.get("systems")
    if not isinstance(systems, list) or not all(isinstance(system, str) and system.strip() for system in systems):
        raise ValueError("OpenRCA preregistration has no valid systems")
    algorithm = prereg.get("algorithm", {})
    current_algorithm = _algorithm_manifest()
    if algorithm.get("source_manifest_sha256") != current_algorithm["sha256"]:
        raise ValueError("OpenRCA algorithm source differs from preregistration")


def _verify_execution_settings(prereg: dict[str, Any], llm: LLMClient, *, no_cache: bool) -> None:
    expected = prereg.get("execution", prereg.get("model", {}))
    expected_model = expected.get("model") or expected.get("name")
    if expected_model and expected_model != "configured runtime model" and expected_model != llm.settings.model:
        raise ValueError(
            f"Runtime model {llm.settings.model!r} does not match preregistration {expected_model!r}"
        )
    if expected.get("temperature") is not None and float(expected["temperature"]) != llm.settings.temperature:
        raise ValueError("Runtime temperature does not match OpenRCA preregistration")
    if expected.get("cache") is False and not no_cache:
        raise ValueError("Frozen OpenRCA run requires --no-cache")


def _verify_prepared_prereg(prepared: PreparedOpenRCA, prereg: dict[str, Any]) -> None:
    expected = prereg.get("dataset", {}).get("prepared")
    if not isinstance(expected, dict):
        raise ValueError("Frozen OpenRCA preregistration does not identify a prepared cache")
    if expected.get("manifest_sha256") != _sha256_file(prepared.manifest_path):
        raise ValueError("Prepared OpenRCA manifest differs from preregistration")


def _execution_fingerprint(
    *,
    dataset: OpenRCADataset,
    prepared: PreparedOpenRCA | None,
    systems: list[str],
    prereg_path: str | None,
    llm: LLMClient | None,
    chunksize: int,
) -> str:
    payload = {
        "query_sha256": _sha256_file(dataset.query_path),
        "prepared_manifest_sha256": _sha256_file(prepared.manifest_path) if prepared else None,
        "systems": systems,
        "prereg_sha256": _sha256_file(Path(prereg_path)) if prereg_path else None,
        "model": llm.settings.model if llm else None,
        "temperature": llm.settings.temperature if llm else None,
        "chunksize": chunksize,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_path(root: str | Path, system: str, row_id: int) -> Path:
    safe_system = "".join(char if char.isalnum() or char in "._-" else "_" for char in system)
    return Path(root) / safe_system / f"row_{row_id:03d}.json"


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("fingerprint") != fingerprint or not isinstance(payload.get("row"), dict):
        return None
    return dict(payload["row"])


def _write_checkpoint(path: Path, fingerprint: str, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"fingerprint": fingerprint, "row": row}, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _volume_bin(prepared: PreparedOpenRCA, row_id: int) -> str:
    for name, ids in prepared.manifest.get("volume_bins", {}).items():
        if row_id in {int(item) for item in ids}:
            return str(name)
    return "unknown"


def _printable_summary(summary_by_system: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for system, rows in summary_by_system.items():
        total = next((row for row in rows if row.get("difficulty") == "total"), rows[-1] if rows else {})
        out[system] = total
    return out


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

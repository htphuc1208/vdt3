"""Run official TeleLogs profile or label-safe LLM evaluations."""
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
from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .dataset import TeleLogsDataset, TeleLogsDatasetError
from .evaluator import score_prediction
from .prereg import DEFAULT_SYSTEMS, _manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run official TeleLogs fallback evaluation.")
    parser.add_argument("--data-dir", default=os.getenv("TELELOGS_DATA_DIR") or "data/telelogs")
    parser.add_argument("--prereg", default=None, help="optional frozen preregistration; fixes split/rows/systems")
    parser.add_argument("--split", default="test")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--row-ids", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--mode", choices=["profile", "llm"], default="profile")
    parser.add_argument("--out", default="results/telelogs_profile.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        dataset = TeleLogsDataset(args.data_dir)
        prereg = _load_prereg(args.prereg) if args.prereg else None
        if prereg:
            _verify_prereg(dataset, prereg)
            split = str(prereg["row_selection"]["split"])
            row_ids = list(prereg["row_selection"]["row_ids"])
            systems = list(prereg["systems"])
        else:
            split = args.split
            row_ids = _parse_row_ids(args.row_ids) if args.row_ids else _first_rows(dataset, split, limit=args.limit or None)
            systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    except (TeleLogsDatasetError, ValueError, OSError, json.JSONDecodeError) as exc:
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "telelogs",
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
        return 2

    if not systems:
        print("ERROR: no systems selected", file=sys.stderr)
        return 2

    llm = None
    if args.mode == "llm":
        settings = get_settings()
        if not settings.has_api_key and not args.cache_only:
            print("ERROR: --mode llm requires OPENAI_API_KEY unless --cache-only is used.", file=sys.stderr)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    rows: list[dict[str, Any]] = []
    for system in systems:
        for task in dataset.iter_runtime_tasks(split, row_ids):
            row = _run_one(dataset, task, split=split, system=system, mode=args.mode, llm=llm)
            rows.append(row)
            print(
                f"[{system}] {split}#{task['row_id']} "
                f"score={row['score']} strict={row['strict_correct']} available={row['score_available']}"
            )

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "telelogs",
            "status": "completed",
            "mode": args.mode,
            "systems": systems,
            "split": split,
            "row_ids": row_ids,
            "prereg": args.prereg,
            "data_dir": str(dataset.root_dir),
            "dataset_manifest_sha256": _manifest(dataset)["sha256"],
            "evidence_warning": _evidence_warning(args.mode),
        },
        "summary": _summary(rows),
        "rows": rows,
    }
    _write_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _run_one(
    dataset: TeleLogsDataset,
    task: dict[str, Any],
    *,
    split: str,
    system: str,
    mode: str,
    llm: LLMClient | None,
) -> dict[str, Any]:
    started = time.time()
    if mode == "profile":
        prediction = _profile_prediction(task, system)
        usage = {"total_tokens": 0, "llm_calls": 0, "tool_calls": 0}
    else:
        assert llm is not None
        prediction, usage = _llm_prediction(task, system, llm)
    raw = dataset.rows(split)[task["row_id"]]
    score = score_prediction(raw, prediction)
    return {
        "case_id": f"{split}:{task['row_id']}",
        "split": split,
        "row_id": task["row_id"],
        "task_id": task["task_id"],
        "system": system,
        "prediction": prediction,
        "latency_s": round(time.time() - started, 3),
        **usage,
        **score.to_dict(),
    }


def _profile_prediction(task: dict[str, Any], system: str) -> dict[str, Any]:
    payload = task.get("payload", {})
    return {
        "root_causes": [],
        "rationale": f"profile-mode TeleLogs smoke for {system}; payload bytes={len(json.dumps(payload, sort_keys=True))}",
    }


def _llm_prediction(task: dict[str, Any], system: str, llm: LLMClient) -> tuple[dict[str, Any], dict[str, int]]:
    if system.startswith("shardrca"):
        return _mas_prediction(task, system, llm)
    return _single_prediction(task, system, llm)


def _single_prediction(task: dict[str, Any], system: str, llm: LLMClient) -> tuple[dict[str, Any], dict[str, int]]:
    system_prompt = (
        "You are one careful 5G network RCA engineer solving an official TeleLogs task. "
        "Use only the label-safe payload. Return ONLY JSON with keys `root_causes` "
        "(array of concise cause labels), `confidence`, and `rationale`."
    )
    if system.endswith("_sc"):
        system_prompt += " Compare plausible causes before finalizing; do not list causes unsupported by evidence."
    resp = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(task, indent=2, sort_keys=True)},
        ],
        force_json=True,
    )
    data = extract_json(resp.content) or {"root_causes": [], "rationale": "LLM response was not valid JSON"}
    return data, _usage_dict(resp.usage)


def _mas_prediction(task: dict[str, Any], system: str, llm: LLMClient) -> tuple[dict[str, Any], dict[str, int]]:
    roles = [
        ("rf_signal", "Focus on RSRP, SINR, PCI collision/mod-30, interference, coverage distance, and antenna geometry."),
        ("mobility_handover", "Focus on UE speed, handover frequency, handover thresholds, and mobility symptoms."),
        ("resource_config", "Focus on throughput, scheduled resource blocks, cell configuration, downtilt, azimuth, and deployment context."),
    ]
    usage = UsageStats()
    findings = []
    for role, instruction in roles:
        resp = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are the {role} specialist in a TeleLogs multi-agent RCA team. {instruction} "
                        "Use only the label-safe payload. Return ONLY JSON with keys `root_causes`, "
                        "`confidence`, `evidence`, and `rationale`."
                    ),
                },
                {"role": "user", "content": json.dumps(task, indent=2, sort_keys=True)},
            ],
            force_json=True,
        )
        usage = usage.add(resp.usage)
        findings.append({"role": role, "finding": extract_json(resp.content) or {}})

    synth_payload = {
        "task": task,
        "specialist_findings": findings,
        "instruction": "Return the final TeleLogs root cause set. Do not include unsupported causes.",
    }
    resp = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are the synthesis agent for a TeleLogs multi-agent RCA team. "
                    "Reconcile specialist findings. Return ONLY JSON with keys `root_causes`, "
                    "`confidence`, `rationale`, and `role_agreement`."
                ),
            },
            {"role": "user", "content": json.dumps(synth_payload, indent=2, sort_keys=True)},
        ],
        force_json=True,
    )
    usage = usage.add(resp.usage)
    data = extract_json(resp.content) or {"root_causes": [], "rationale": "synthesis response was not valid JSON"}
    data["mas_role_outputs"] = findings
    return data, _usage_dict(usage)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    out = aggregate_ci([
        {
            "system": row["system"],
            "strict_correct": bool(row.get("strict_correct")),
            "score": float(row.get("score") or 0.0),
            "score_available": bool(row.get("score_available")),
            "total_tokens": int(row.get("total_tokens") or 0),
            "llm_calls": int(row.get("llm_calls") or 0),
        }
        for row in rows
    ], ["strict_correct", "score", "score_available", "total_tokens", "llm_calls"])
    by_split: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_split[row["system"]][row["split"]].append(float(row.get("score") or 0.0))
    for system, splits in by_split.items():
        out.setdefault(system, {})
        out[system]["per_split_score"] = {
            name: round(sum(values) / len(values), 4)
            for name, values in sorted(splits.items())
        }
    return out


def _usage_dict(usage: UsageStats) -> dict[str, int]:
    return {
        "total_tokens": usage.total_tokens,
        "llm_calls": usage.llm_calls,
        "tool_calls": usage.tool_calls,
    }


def _evidence_warning(mode: str) -> str:
    if mode == "profile":
        return "profile mode is only an ingestion/scoring smoke test"
    return "TeleLogs is an official synthetic 5G RCA fallback; report below real OpenRCA/TN-RCA evidence"


def _first_rows(dataset: TeleLogsDataset, split: str, *, limit: int | None) -> list[int]:
    ids = list(range(len(dataset.rows(split))))
    return ids if limit is None else ids[: min(limit, len(ids))]


def _load_prereg(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_prereg(dataset: TeleLogsDataset, prereg: dict[str, Any]) -> None:
    if prereg.get("status") != "frozen":
        raise ValueError("TeleLogs preregistration is not frozen")
    manifest = _manifest(dataset)
    if prereg.get("dataset", {}).get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("TeleLogs dataset manifest differs from preregistration")
    split = prereg.get("row_selection", {}).get("split")
    row_ids = prereg.get("row_selection", {}).get("row_ids")
    if not isinstance(split, str) or split not in dataset.splits:
        raise ValueError("TeleLogs preregistration split is unavailable")
    if not isinstance(row_ids, list) or not row_ids:
        raise ValueError("TeleLogs preregistration has no row_ids")
    row_count = len(dataset.rows(split))
    if any(not isinstance(row_id, int) or row_id < 0 or row_id >= row_count for row_id in row_ids):
        raise ValueError("TeleLogs preregistration contains invalid row_ids")
    systems = prereg.get("systems")
    if not isinstance(systems, list) or not all(isinstance(system, str) and system.strip() for system in systems):
        raise ValueError("TeleLogs preregistration has no valid systems")


def _parse_row_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

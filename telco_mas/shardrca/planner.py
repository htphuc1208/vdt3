"""Shard planning for ShardRCA."""
from __future__ import annotations

import json
from typing import Any

from ..llm import LLMClient, extract_json
from .catalog import ShardSpec, TelemetryCatalog, make_component_group_shards, make_default_shards


PLANNER_SYSTEM = """You are the ShardRCA planner.
Given a label-safe task payload and a telemetry catalog, return a compact JSON shard plan.
Use disjoint shards by modality and component groups. Do not request labels.
Return ONLY JSON:
{"shards": [{"shard_id": "metrics_0", "modality": "metrics",
             "paths": ["..."], "components": [], "rationale": "..."}]}"""


def plan_shards(
    catalog: TelemetryCatalog,
    *,
    task_payload: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    max_shards: int = 3,
) -> list[ShardSpec]:
    """Return a validated shard plan, falling back to deterministic modality shards."""

    fallback = (
        make_component_group_shards(catalog, max_groups=max(1, max_shards // 3))
        if str(catalog.dataset).startswith("RE")
        else make_default_shards(catalog, max_shards=max_shards)
    )
    if llm is None:
        return fallback

    user_payload = {
        "task": task_payload or {"case_id": catalog.case_id},
        "catalog": catalog.summary(),
        "files": [
            {
                "path": item.path,
                "relative_path": item.relative_path,
                "modality": item.modality,
                "bytes": item.size_bytes,
                "rows": item.row_count,
                "time_range": [item.start_time, item.end_time],
                "columns_preview": item.columns_preview[:12],
            }
            for item in catalog.files
        ],
    }
    try:
        response = llm.chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, default=str)},
            ],
            force_json=True,
        )
        data = extract_json(response.content)
        shards = []
        known_paths = {item.path for item in catalog.files}
        for idx, raw in enumerate(data.get("shards", []) or []):
            paths = [str(path) for path in raw.get("paths", []) if str(path) in known_paths]
            modality = str(raw.get("modality") or "").strip()
            if not paths or modality not in {"metrics", "logs", "traces"}:
                continue
            shards.append(
                ShardSpec(
                    shard_id=str(raw.get("shard_id") or f"{modality}_{idx}"),
                    modality=modality,
                    paths=paths,
                    query_time=catalog.query_time,
                    start_time=catalog.time_range_start,
                    end_time=catalog.time_range_end,
                    components=[str(c) for c in raw.get("components", []) or []],
                    metadata={"planner": "llm", "rationale": str(raw.get("rationale") or "")},
                )
            )
        return shards[:max_shards] or fallback
    except Exception:
        return fallback

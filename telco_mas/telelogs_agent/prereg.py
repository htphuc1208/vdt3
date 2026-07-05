"""Pre-registration generator for the TeleLogsAgent synthetic 5G fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import TeleLogsAgentDataset, TeleLogsAgentDatasetError


DEFAULT_SYSTEMS = [
    "single_react",
    "single_react_sc",
    "single_equal_tokens",
    "code_retrieval_single",
    "same_board_single",
    "shardrca_full",
]


def build_preregistration(
    dataset: TeleLogsAgentDataset,
    *,
    systems: list[str] | None = None,
    limit_per_set: int | None = None,
    seed: int = 12,
    model: str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    selected = _select_rows(dataset, limit_per_set=limit_per_set, seed=seed)
    manifest = _manifest(dataset)
    return {
        "status": "frozen",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": {
            "name": "TeleLogsAgent",
            "type": "synthetic 5G tool-use RCA fallback",
            "role_in_claim": "fallback/secondary evidence only unless real OpenRCA/TN-RCA data is unavailable",
            "source": "https://huggingface.co/datasets/netop/TeleLogsAgent",
        },
        "dataset": {
            "root_dir": str(dataset.root_dir),
            "scenario_sets": list(dataset.scenario_sets),
            "counts": dataset.counts(),
            "file_manifest": manifest["files"],
            "manifest_sha256": manifest["sha256"],
            "total_bytes": manifest["total_bytes"],
        },
        "row_selection": {
            "method": "all_rows" if limit_per_set is None else "per_set_random_sample",
            "seed": seed,
            "limit_per_set": limit_per_set,
            "selected": selected,
            "total_selected": sum(len(ids) for ids in selected.values()),
        },
        "systems": systems or DEFAULT_SYSTEMS,
        "model": {
            "name": model or os.getenv("OPENAI_MODEL") or "configured runtime model",
            "temperature": temperature,
            "cache": False,
        },
        "runtime_safety": {
            "runtime_task_fields": ["scenario_set", "row_id", "task_id", "payload"],
            "label_like_keys_removed_from_payload": [
                "answer",
                "correct",
                "expected",
                "ground_truth",
                "label",
                "root_cause",
                "scoring",
                "solution",
                "target",
            ],
            "forbidden_runtime_inputs": ["gold answer", "root cause label", "post-hoc scenario filtering"],
        },
        "metrics": {
            "primary": "official_telelogsagent_task_success",
            "secondary": [
                "per_TS_accuracy",
                "tool_calls",
                "tool_failure_rate",
                "tool_call_efficiency",
                "llm_calls",
                "tokens",
                "latency_s",
            ],
        },
        "clear_win_gate": {
            "primary_effect": "MAS beats strongest operational single baseline by >=0.10 absolute accuracy or >=20% relative error reduction",
            "significance": "paired exact p <= 0.05 when sample size permits",
            "diagnostics_required": ["same_board_single", "compute accounting", "per_TS breakdown"],
        },
        "stopping_rule": {
            "fixed_rows": True,
            "no_extension_by_p_value": True,
            "no_post_hoc_filtering": True,
            "official_test_split_only_for_final": True,
        },
        "note": (
            "TeleLogsAgent is synthetic. Use this only as a telecom-valid fallback/secondary benchmark, "
            "and disclose that real OpenRCA/TN-RCA evidence is still preferred."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a TeleLogsAgent fallback preregistration JSON.")
    parser.add_argument("--data-dir", default=os.getenv("TELELOGS_AGENT_DATA_DIR") or "data/telelogs_agent")
    parser.add_argument("--out", default="results/prereg_telelogs_agent_frozen.json")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--limit-per-set", type=int, default=0, help="0 means all rows")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args(argv)

    try:
        dataset = TeleLogsAgentDataset(args.data_dir)
        payload = build_preregistration(
            dataset,
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
            limit_per_set=args.limit_per_set or None,
            seed=args.seed,
            model=args.model,
            temperature=args.temperature,
        )
    except (TeleLogsAgentDatasetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "selected": payload["row_selection"]["selected"],
        "manifest_sha256": payload["dataset"]["manifest_sha256"],
    }, indent=2))
    return 0


def _select_rows(dataset: TeleLogsAgentDataset, *, limit_per_set: int | None, seed: int) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    rng = random.Random(seed)
    for name in dataset.scenario_sets:
        ids = list(range(len(dataset.rows(name))))
        if limit_per_set is not None:
            if limit_per_set <= 0:
                raise ValueError("limit_per_set must be positive when supplied")
            rng.shuffle(ids)
            ids = sorted(ids[: min(limit_per_set, len(ids))])
        selected[name] = ids
    return selected


def _manifest(dataset: TeleLogsAgentDataset) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for name in dataset.scenario_sets:
        path = dataset.test_path(name)
        size = path.stat().st_size
        total_bytes += size
        files.append({
            "scenario_set": name,
            "path": str(path.relative_to(dataset.root_dir)),
            "bytes": size,
            "sha256": _sha256_file(path),
        })
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files": files,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "total_bytes": total_bytes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

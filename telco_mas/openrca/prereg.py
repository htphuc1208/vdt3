"""Pre-registration generator for OpenRCA Telecom experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import OpenRCADataset, OpenRCADatasetError
from .prepared import PreparedOpenRCA, PreparedOpenRCAError
from .sandbox import source_image_tag


DEFAULT_SYSTEMS = [
    "single_react_sc",
    "rca_agent_replica",
    "shardrca_full",
    "no_falsifier",
    "no_topology",
    "no_interaction",
    "no_refinement",
]


def build_preregistration(
    dataset: OpenRCADataset,
    *,
    systems: list[str] | None = None,
    row_ids: list[int] | None = None,
    limit: int | None = None,
    seed: int = 12,
    model: str | None = None,
    temperature: float = 0.0,
    prepared: PreparedOpenRCA | None = None,
    algorithm_id: str | None = None,
    contaminated_row_ids: list[int] | None = None,
) -> dict[str, Any]:
    selected = _validate_row_ids(dataset, row_ids) if row_ids is not None else _select_row_ids(dataset, limit=limit, seed=seed)
    tasks = [dataset.get_runtime_task(row_id) for row_id in selected]
    difficulties = Counter(_difficulty(task["task_index"]) for task in tasks)
    query_hash = _sha256_file(dataset.query_path)
    telemetry_manifest = _telemetry_manifest(dataset.telemetry_dir)
    row_id_text = ",".join(str(row_id) for row_id in selected)
    selected_systems = systems or DEFAULT_SYSTEMS
    algorithm = _algorithm_manifest()
    resolved_algorithm_id = algorithm_id or algorithm["sha256"]
    prepared_payload = None
    volume_bins: dict[str, list[int]] = {}
    if prepared is not None:
        prepared.validate_against(dataset)
        prepared_payload = {
            "root": str(prepared.root),
            "manifest_path": str(prepared.manifest_path),
            "manifest_sha256": _sha256_file(prepared.manifest_path),
            "format_version": prepared.manifest.get("format_version"),
            "row_count": prepared.manifest.get("row_count"),
        }
        selected_set = set(selected)
        volume_bins = {
            name: [int(row_id) for row_id in ids if int(row_id) in selected_set]
            for name, ids in prepared.manifest.get("volume_bins", {}).items()
        }
    smoke_ids = _smoke_row_ids(tasks)
    contaminated = sorted(set(contaminated_row_ids or []))
    invalid_contaminated = [row_id for row_id in contaminated if row_id not in set(selected)]
    if invalid_contaminated:
        raise ValueError(f"contaminated row IDs are outside the selected rows: {invalid_contaminated}")
    confirmatory_rows = [row_id for row_id in selected if row_id not in set(contaminated)]
    return {
        "status": "frozen",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": f"OpenRCA {dataset.dataset}",
            "root_dir": str(dataset.root_dir),
            "query_path": str(dataset.query_path),
            "telemetry_dir": str(dataset.telemetry_dir),
            "query_sha256": query_hash,
            "telemetry_manifest_sha256": telemetry_manifest["sha256"],
            "telemetry_file_count": telemetry_manifest["file_count"],
            "telemetry_total_bytes": telemetry_manifest["total_bytes"],
            "telemetry_date_dirs": dataset.available_dates(limit=1000),
            "prepared": prepared_payload,
        },
        "row_selection": {
            "method": "explicit_row_ids" if row_ids is not None else ("all_rows" if limit is None or limit >= len(dataset.rows) else "stratified_balanced"),
            "seed": seed,
            "row_count": len(selected),
            "row_ids": selected,
            "row_ids_arg": row_id_text,
            "difficulty_counts": dict(sorted(difficulties.items())),
            "confirmatory_row_ids": confirmatory_rows,
            "confirmatory_row_count": len(confirmatory_rows),
        },
        "contamination_ledger": [
            {
                "row_id": row_id,
                "status": "engineering_only",
                "reason": (
                    "OpenRCA evaluator output for an offline deterministic plumbing run was observed "
                    "before preregistration; the row remains in the 51-row report but is excluded "
                    "from confirmatory gates."
                ),
            }
            for row_id in contaminated
        ],
        "systems": selected_systems,
        "system_roles": {
            "operational_single": ["single_react_sc"],
            "architecture_baseline": ["rca_agent_replica"],
            "treatment": "shardrca_full",
        },
        "algorithm": {
            "id": resolved_algorithm_id,
            "source_manifest_sha256": algorithm["sha256"],
            "files": algorithm["files"],
        },
        "overfit_guard": {
            "fitted_weights_allowed": False,
            "fusion_weights": "default_no_fit",
            "no_post_hoc_weight_fit": True,
            "no_post_hoc_row_filtering": True,
            "required_ablations": [
                "no_interaction",
                "no_falsifier",
                "no_topology",
                "no_refinement",
            ],
            "required_analysis_checks": [
                "weights_declared_no_fit",
                "candidate_catalog_not_label_derived",
                "strongest_single_baseline",
                "architecture_baseline_comparison",
                "operational_single_comparison",
            ],
        },
        "model": {
            "name": model or os.getenv("OPENAI_MODEL") or "configured runtime model",
            "temperature": temperature,
            "cache": False,
        },
        "execution": {
            "model": model or os.getenv("OPENAI_MODEL") or "configured runtime model",
            "temperature": temperature,
            "cache": False,
            "case_order": "smoke_gate_then_row_id",
            "system_order": "deterministic_row_rotation",
            "case_concurrency": 1,
            "worker_concurrency": 3,
            "chunksize": 100_000,
            "budgets": {
                "rca_agent_replica": {
                    "controller_max_steps": 25,
                    "executor_attempts_per_step": 2,
                    "case_timeout_s": 600,
                    "sandbox_cpu": 2,
                    "sandbox_memory_gb": 3,
                    "sandbox_image": source_image_tag(),
                },
                "single_react_sc": {"runs": 3, "tool_calls_per_run": 3},
                "shardrca_full": {"workers": 5, "falsifier_calls": 1},
                "no_falsifier": {"workers": 5, "falsifier_calls": 0},
                "no_topology": {"workers": 5, "topology_temporal_rerank": False},
                "no_interaction": {"workers": 5, "autonomous_peer_interaction": False},
                "no_refinement": {"workers": 5, "iterative_refinement": False},
            },
        },
        "smoke_gate": {
            "row_ids": smoke_ids,
            "evaluation_blinded_until_full_completion": True,
            "on_failure": "invalidate run if code or prompts change; create a new preregistration",
        },
        "volume_bins": volume_bins,
        "runtime_safety": {
            "runtime_task_fields": ["row_id", "task_index", "instruction"],
            "evaluator_only_fields": ["scoring_points"],
            "forbidden_runtime_inputs": ["scoring_points", "true root cause", "post-hoc row filtering"],
        },
        "metrics": {
            "primary": "strict_openrca_accuracy",
            "primary_definition": "prediction receives full score from the OpenRCA evaluator",
            "secondary": [
                "partial_openrca_score",
                "component_accuracy_when_requested",
                "reason_accuracy_when_requested",
                "time_accuracy_when_requested",
                "tokens",
                "tool_calls",
                "llm_calls",
                "latency_s",
            ],
        },
        "statistical_tests": [
            "paired exact sign or McNemar test for strict correctness",
            "paired bootstrap confidence interval for partial score delta",
        ],
        "clear_win_gate": {
            "primary_effect": (
                "shardrca_full beats both single_react_sc and rca_agent_replica by >=0.10 "
                "absolute strict accuracy or >=20% relative error reduction"
            ),
            "significance": "Holm-adjusted paired exact p <= 0.05 for both confirmatory comparisons",
            "strong_mechanism": "also has a positive high-volume-bin delta",
            "diagnostics_required": [
                "high_volume_bin",
                "tokens",
                "tool_calls",
                "llm_calls",
                "latency_s",
                "no_falsifier",
                "no_topology",
                "no_interaction",
                "no_refinement",
            ],
        },
        "stopping_rule": {
            "fixed_rows": True,
            "no_extension_by_p_value": True,
            "no_post_hoc_filtering": True,
        },
        "commands": [
            "python3 -m telco_mas.openrca.cli "
            "--mode llm --confirm-live-llm --prereg results/prereg_openrca_telecom_frozen.json "
            f"--systems {','.join(selected_systems)} --row-ids {row_id_text} "
            "--prepared-dir data/openrca_prepared/Telecom "
            "--checkpoint-dir results/checkpoints/openrca_telecom_frozen "
            "--resume --no-cache --out results/openrca_paired_frozen.json"
        ],
        "analysis_command": (
            "python3 -m telco_mas.openrca.result_analysis results/openrca_paired_frozen.json "
            "--baseline strongest_single --treatment shardrca_full "
            "--out results/openrca_paired_frozen_analysis.json"
        ),
        "note": "Generated before live evaluation. Regenerate only if dataset files, row selection, systems, or budgets intentionally change.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze an OpenRCA Telecom preregistration JSON.")
    parser.add_argument("--data-dir", default=os.getenv("OPENRCA_DATA_DIR") or "data/openrca")
    parser.add_argument("--dataset", default="Telecom")
    parser.add_argument("--out", default="results/prereg_openrca_telecom_frozen.json")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--row-ids", default=None, help="comma-separated row IDs to freeze exactly")
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows; otherwise balanced stratified sample")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prepared-dir", default="data/openrca_prepared/Telecom")
    parser.add_argument("--algorithm-id", default=None)
    parser.add_argument("--contaminated-row-ids", default=None)
    args = parser.parse_args(argv)

    try:
        dataset = OpenRCADataset(args.data_dir, dataset=args.dataset)
        prepared = PreparedOpenRCA(args.prepared_dir)
        payload = build_preregistration(
            dataset,
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
            row_ids=_parse_row_ids(args.row_ids) if args.row_ids else None,
            limit=args.limit or None,
            seed=args.seed,
            model=args.model,
            temperature=args.temperature,
            prepared=prepared,
            algorithm_id=args.algorithm_id,
            contaminated_row_ids=(
                _parse_contaminated_row_ids(args.contaminated_row_ids, dataset)
                if args.contaminated_row_ids
                else None
            ),
        )
    except (OpenRCADatasetError, PreparedOpenRCAError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "row_count": payload["row_selection"]["row_count"],
        "difficulty_counts": payload["row_selection"]["difficulty_counts"],
        "query_sha256": payload["dataset"]["query_sha256"],
        "telemetry_manifest_sha256": payload["dataset"]["telemetry_manifest_sha256"],
    }, indent=2))
    return 0


def _select_row_ids(dataset: OpenRCADataset, *, limit: int | None, seed: int) -> list[int]:
    row_count = len(dataset.rows)
    if limit is None or limit >= row_count:
        return list(range(row_count))
    if limit <= 0:
        raise ValueError("limit must be positive when supplied")

    groups: dict[str, list[int]] = defaultdict(list)
    for row_id, row in enumerate(dataset.rows):
        groups[_difficulty(str(row.get("task_index") or ""))].append(row_id)
    rng = random.Random(seed)
    for ids in groups.values():
        rng.shuffle(ids)

    selected: list[int] = []
    order = ["easy", "middle", "hard"]
    while len(selected) < limit and any(groups.values()):
        for name in order:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
    return sorted(selected)


def _validate_row_ids(dataset: OpenRCADataset, row_ids: list[int]) -> list[int]:
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique")
    for row_id in row_ids:
        if row_id < 0 or row_id >= len(dataset.rows):
            raise ValueError(f"row_id {row_id} outside query.csv range 0..{len(dataset.rows) - 1}")
    return list(row_ids)


def _parse_contaminated_row_ids(text: str, dataset: OpenRCADataset) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(len(dataset.rows)))
    return _parse_row_ids(text)


def _telemetry_manifest(root: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(path.relative_to(root)), "bytes": size})
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difficulty(task_index: str) -> str:
    try:
        task_id = int(str(task_index).split("_")[1])
    except (IndexError, ValueError):
        return "hard"
    if task_id <= 3:
        return "easy"
    if task_id <= 6:
        return "middle"
    return "hard"


def _parse_row_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _smoke_row_ids(tasks: list[dict[str, Any]]) -> list[int]:
    by_difficulty: dict[str, int] = {}
    for task in tasks:
        difficulty = _difficulty(str(task["task_index"]))
        by_difficulty.setdefault(difficulty, int(task["row_id"]))
    return [by_difficulty[name] for name in ("easy", "middle", "hard") if name in by_difficulty]


def _algorithm_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = [
        "telco_mas/openrca/task_parser.py",
        "telco_mas/openrca/prepared.py",
        "telco_mas/openrca/workers.py",
        "telco_mas/openrca/rca_agent_replica.py",
        "telco_mas/openrca/sandbox_kernel.py",
        "telco_mas/shardrca/runner.py",
        "telco_mas/shardrca/fusion.py",
        "telco_mas/shardrca/single_baseline.py",
    ]
    files = []
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": files}


if __name__ == "__main__":
    raise SystemExit(main())

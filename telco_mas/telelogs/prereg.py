"""Pre-registration generator for the official TeleLogs 5G RCA fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import TeleLogsDataset, TeleLogsDatasetError


DEFAULT_SYSTEMS = ["single_react_sc", "shardrca_full"]


def build_preregistration(
    dataset: TeleLogsDataset,
    *,
    split: str = "test",
    systems: list[str] | None = None,
    row_ids: list[int] | None = None,
    limit: int | None = None,
    seed: int = 17,
    model: str | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    selected = _validate_row_ids(dataset, split, row_ids) if row_ids is not None else _select_row_ids(
        dataset,
        split=split,
        limit=limit,
        seed=seed,
    )
    manifest = _manifest(dataset)
    selected_systems = systems or DEFAULT_SYSTEMS
    row_ids_arg = ",".join(str(row_id) for row_id in selected)
    return {
        "status": "frozen",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": {
            "name": "TeleLogs",
            "source": "https://huggingface.co/datasets/netop/TeleLogs",
            "role_in_claim": "official synthetic 5G RCA fallback, below real OpenRCA/TN-RCA evidence",
        },
        "dataset": {
            "root_dir": str(dataset.root_dir),
            "splits": list(dataset.splits),
            "counts": dataset.counts(),
            "manifest_sha256": manifest["sha256"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
        },
        "row_selection": {
            "split": split,
            "method": "explicit_row_ids" if row_ids is not None else ("all_rows" if limit is None else "seeded_sample"),
            "seed": seed,
            "row_count": len(selected),
            "row_ids": selected,
            "row_ids_arg": row_ids_arg,
        },
        "systems": selected_systems,
        "model": {
            "name": model or os.getenv("OPENAI_MODEL") or "configured runtime model",
            "temperature": temperature,
            "cache": False,
        },
        "runtime_safety": {
            "runtime_task_fields": ["split", "row_id", "task_id", "payload"],
            "label_like_keys_removed_from_payload": [
                "answer",
                "correct",
                "expected",
                "ground_truth",
                "label",
                "root_cause",
                "root_causes",
                "scoring",
                "solution",
                "target",
            ],
            "forbidden_runtime_inputs": ["root cause label", "answer", "post-hoc row filtering"],
        },
        "metrics": {
            "primary": "strict_root_cause_set_accuracy",
            "secondary": ["jaccard_root_cause_score", "tokens", "llm_calls", "latency_s"],
        },
        "clear_win_gate": {
            "primary_effect": "MAS beats strongest operational single baseline by >=0.10 absolute strict accuracy or >=20% relative error reduction",
            "significance": "paired exact p <= 0.05 when sample size permits",
            "diagnostics_required": ["score", "tokens", "llm_calls", "per_split_summary"],
        },
        "stopping_rule": {
            "fixed_rows": True,
            "no_extension_by_p_value": True,
            "no_post_hoc_filtering": True,
            "official_test_split_only_for_final": split == "test",
        },
        "commands": [
            "python3 -m telco_mas.telelogs.cli "
            f"--mode llm --prereg results/prereg_telelogs_frozen.json --out results/telelogs_paired_llm.json"
        ],
        "analysis_command": (
            "python3 -m telco_mas.telelogs.result_analysis results/telelogs_paired_llm.json "
            "--baseline strongest_single --treatment shardrca_full "
            "--out results/telelogs_paired_llm_analysis.json"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a TeleLogs fallback preregistration JSON.")
    parser.add_argument("--data-dir", default=os.getenv("TELELOGS_DATA_DIR") or "data/telelogs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--row-ids", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--out", default="results/prereg_telelogs_frozen.json")
    args = parser.parse_args(argv)

    try:
        dataset = TeleLogsDataset(args.data_dir)
        payload = build_preregistration(
            dataset,
            split=args.split,
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
            row_ids=_parse_row_ids(args.row_ids) if args.row_ids else None,
            limit=args.limit or None,
            seed=args.seed,
            model=args.model,
            temperature=args.temperature,
        )
    except (TeleLogsDatasetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "split": payload["row_selection"]["split"],
        "row_count": payload["row_selection"]["row_count"],
        "manifest_sha256": payload["dataset"]["manifest_sha256"],
    }, indent=2))
    return 0


def _select_row_ids(dataset: TeleLogsDataset, *, split: str, limit: int | None, seed: int) -> list[int]:
    row_count = len(dataset.rows(split))
    if limit is None or limit >= row_count:
        return list(range(row_count))
    if limit <= 0:
        raise ValueError("limit must be positive when supplied")
    ids = list(range(row_count))
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:limit])


def _validate_row_ids(dataset: TeleLogsDataset, split: str, row_ids: list[int]) -> list[int]:
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique")
    row_count = len(dataset.rows(split))
    for row_id in row_ids:
        if row_id < 0 or row_id >= row_count:
            raise ValueError(f"row_id {row_id} outside {split} range 0..{row_count - 1}")
    return list(row_ids)


def _manifest(dataset: TeleLogsDataset) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in dataset.json_paths():
        size = path.stat().st_size
        total_bytes += size
        files.append({
            "path": str(path.relative_to(dataset.root_dir)),
            "bytes": size,
            "sha256": _sha256_file(path),
        })
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_row_ids(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

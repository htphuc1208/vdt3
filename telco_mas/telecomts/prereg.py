"""Pre-registration for a source-session-held-out TelecomTS RCA experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .catalog import CATALOG_POLICY_ID, build_training_catalog, catalog_sha256
from .dataset import RCA_CLASSES, SOURCE_URL, SPLIT_POLICY_ID, TelecomTSDataset, TelecomTSDatasetError


DEFAULT_SYSTEMS = [
    "single_react",
    "single_react_sc",
    "single_equal_calls",
    "same_board_single",
    "telecomts_shardrca_full",
]


def build_preregistration(
    dataset: TelecomTSDataset,
    *,
    split: str = "test",
    systems: list[str] | None = None,
    limit_per_class: int | None = 10,
    seed: int = 20260704,
    model: str | None = None,
    temperature: float = 0.1,
    runs: int = 1,
    freeze: bool = False,
    algorithm_id: str | None = None,
) -> dict[str, Any]:
    if freeze and not algorithm_id:
        raise ValueError("--freeze requires a non-empty --algorithm-id")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    selected = _select_event_indices(
        dataset,
        split=split,
        limit_per_class=limit_per_class,
        seed=seed,
    )
    events = dataset.events(split)
    selected_case_ids = [events[event_index].event_id for event_index in selected]
    selected_counts = defaultdict(int)
    for event_index in selected:
        selected_counts[events[event_index].root_cause] += 1
    manifest = dataset_manifest(dataset)
    training_catalog = build_training_catalog(dataset)
    return {
        "status": "frozen" if freeze else "draft",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": {
            "name": "TelecomTS",
            "source": SOURCE_URL,
            "paper": "https://arxiv.org/abs/2510.06063",
            "task": "ten-class root-cause classification over merged anomalous 5G KPI events",
            "evidence_tier": "public_5g_testbed_backed_synthetic_rca",
            "role_in_claim": "synthetic-only fallback; not real-fault or operator-network RCA",
            "critical_caveat": (
                "The upstream RCA task excludes real over-the-air jamming and classifies ten "
                "literature-grounded synthetic anomaly types injected into measured testbed KPIs."
            ),
        },
        "dataset": {
            "root_dir": str(dataset.root_dir),
            "manifest_sha256": manifest["sha256"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "counts": dataset.counts(),
            "class_counts": dataset.class_counts(),
            "event_counts": dataset.event_counts(),
            "class_event_counts": dataset.class_event_counts(),
            "source_sessions": list(dataset.source_sessions),
            "complete_official_layout": dataset.complete_official_layout,
        },
        "split_policy": {
            "id": SPLIT_POLICY_ID,
            "unit": "source session (zone x application), never individual windows",
            "development": ["Zone_A/File", "Zone_B/Twitch", "Zone_C/YouTube"],
            "validation": ["Zone_A/Twitch", "Zone_B/YouTube", "Zone_C/File"],
            "test": ["Zone_A/YouTube", "Zone_B/File", "Zone_C/Twitch"],
            "reason": "prevent adjacent-window and same-session leakage across development and test",
        },
        "training_catalog": {
            "policy_id": CATALOG_POLICY_ID,
            "source_split": "development",
            "sha256": catalog_sha256(training_catalog),
            "class_count": len(training_catalog),
            "uses_validation_or_test_labels": False,
        },
        "event_selection": {
            "split": split,
            "unit": "merged anomaly event; overlapping stride-32 windows are not independent cases",
            "method": "seeded_class_cap" if limit_per_class is not None else "all_split_events",
            "seed": seed,
            "limit_per_class": limit_per_class,
            "event_count": len(selected),
            "event_indices": selected,
            "event_ids": selected_case_ids,
            "class_counts": {name: selected_counts[name] for name in RCA_CLASSES},
        },
        "systems": systems or DEFAULT_SYSTEMS,
        "algorithm": {
            "id": algorithm_id or "unlocked draft; runner and prompts may still change",
            "locked_before_test": bool(freeze),
        },
        "execution": {
            "runs": runs,
            "model": model or os.getenv("OPENAI_MODEL") or "configured runtime model",
            "temperature": temperature,
            "cache": False,
        },
        "runtime_safety": {
            "visible": [
                "opaque case ID",
                "overlap-deduplicated relative KPI arrays for one anomaly event",
                "sampling rate",
                "zone/application/mobility/congestion context",
                "fixed candidate root-cause universe",
            ],
            "withheld": [
                "anomalies object and type",
                "affected KPI labels",
                "troubleshooting ticket",
                "Q&A answers and reasoning",
                "generated description and statistics",
                "absolute timestamps and source path",
                "anomaly_present label",
            ],
        },
        "metrics": {
            "primary": "macro_root_cause_accuracy",
            "secondary": [
                "micro_root_cause_accuracy",
                "per_class_accuracy",
                "paired_exact_mcnemar",
                "tokens",
                "llm_calls",
                "tool_calls",
                "latency_s",
            ],
        },
        "clear_win_gate": {
            "baseline": "strongest operational single selected from frozen systems",
            "effect": "macro accuracy delta >= 0.10",
            "significance": "paired exact McNemar p <= 0.05 on row correctness",
            "compute": (
                "single_equal_calls receives five full-board calls; exact token matching is unsupported, "
                "so report measured tokens and do not label it equal-token"
            ),
        },
        "stopping_rule": {
            "fixed_events": True,
            "no_extension_by_p_value": True,
            "no_post_hoc_row_or_class_filtering": True,
            "test_used_once_after_algorithm_lock": split == "test" and freeze,
        },
    }


def dataset_manifest(dataset: TelecomTSDataset) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in dataset.jsonl_paths():
        size = path.stat().st_size
        total_bytes += size
        files.append({
            "path": path.relative_to(dataset.root_dir).as_posix(),
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


def _select_event_indices(
    dataset: TelecomTSDataset,
    *,
    split: str,
    limit_per_class: int | None,
    seed: int,
) -> list[int]:
    events = dataset.events(split)
    if limit_per_class is not None and limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive when supplied")
    by_class: dict[str, list[int]] = defaultdict(list)
    for event_index, event in enumerate(events):
        by_class[event.root_cause].append(event_index)
    missing = [name for name in RCA_CLASSES if not by_class[name]]
    if missing:
        raise ValueError(f"split {split} is missing RCA classes: {', '.join(missing)}")
    selected = []
    for name in RCA_CLASSES:
        ids = list(by_class[name])
        random.Random(f"{seed}:{name}").shuffle(ids)
        selected.extend(ids if limit_per_class is None else ids[:limit_per_class])
    return sorted(selected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a TelecomTS RCA preregistration.")
    parser.add_argument("--data-dir", default=os.getenv("TELECOMTS_DATA_DIR") or "data/telecomts")
    parser.add_argument("--split", choices=["development", "validation", "test"], default="test")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--limit-per-class", type=int, default=10, help="0 selects all rows")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--algorithm-id", default=None)
    parser.add_argument("--out", default="results/prereg_telecomts_draft.json")
    args = parser.parse_args(argv)

    try:
        dataset = TelecomTSDataset(args.data_dir)
        payload = build_preregistration(
            dataset,
            split=args.split,
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
            limit_per_class=args.limit_per_class or None,
            seed=args.seed,
            model=args.model,
            temperature=args.temperature,
            runs=args.runs,
            freeze=args.freeze,
            algorithm_id=args.algorithm_id,
        )
    except (TelecomTSDatasetError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "status": payload["status"],
        "split": payload["event_selection"]["split"],
        "event_count": payload["event_selection"]["event_count"],
        "class_counts": payload["event_selection"]["class_counts"],
        "manifest_sha256": payload["dataset"]["manifest_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

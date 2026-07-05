"""Build a label-safe RCAEval hard split for ShardRCA experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import median
from typing import Any

from ..evaluation.rcaeval_adapter import _case_from_dir, iter_case_dirs, resolve_rcaeval_root, validate_rcaeval


def build_hard_split(
    root: str | None = None,
    *,
    min_criteria: int = 3,
    wide_metric_columns_min: int = 120,
) -> dict[str, Any]:
    """Select hard RCAEval cases using label-free metadata only."""

    root_path = resolve_rcaeval_root(root)
    case_dirs = list(iter_case_dirs(root_path))
    sizes = {case_dir: _case_size(case_dir) for case_dir in case_dirs}
    size_median = median(sizes.values()) if sizes else 0
    rows = []
    for case_dir in case_dirs:
        row = _hard_row(root_path, case_dir, size_bytes=sizes[case_dir], size_median=size_median,
                        wide_metric_columns_min=wide_metric_columns_min)
        score = row["criteria_score"]
        if score >= min_criteria:
            rows.append(row)
    rows.sort(key=lambda row: (-row["criteria_score"], -row["telemetry_bytes"], row["runtime_case_id"]))
    return {
        "meta": {
            "suite": "rcaeval_hard",
            "selection": "label-free hard split over RCAEval metadata",
            "min_criteria": min_criteria,
            "wide_metric_columns_min": wide_metric_columns_min,
            "telemetry_median_bytes": size_median,
            "validation": validate_rcaeval(root),
            "label_safety": (
                "runtime_case_id is opaque; ground-truth root/fault labels and label-derived case paths "
                "are intentionally omitted from this split artifact"
            ),
        },
        "cases": rows,
    }


def load_hard_cases(
    root: str | None = None,
    *,
    sample: int | None = None,
    seed: int = 7,
    min_criteria: int = 3,
    wide_metric_columns_min: int = 120,
):
    """Load selected hard cases, building expensive observability only for selected dirs."""

    root_path = resolve_rcaeval_root(root)
    case_dirs = list(iter_case_dirs(root_path))
    sizes = {case_dir: _case_size(case_dir) for case_dir in case_dirs}
    size_median = median(sizes.values()) if sizes else 0
    selected = [
        case_dir
        for case_dir in case_dirs
        if _hard_row(
            root_path,
            case_dir,
            size_bytes=sizes[case_dir],
            size_median=size_median,
            wide_metric_columns_min=wide_metric_columns_min,
        )["criteria_score"] >= min_criteria
    ]
    if sample and sample > 0 and sample < len(selected):
        rng = random.Random(seed)
        selected = rng.sample(selected, sample)
    return sorted((_case_from_dir(case_dir) for case_dir in selected), key=lambda case: case.case_id)


def load_hard_cases_by_runtime_ids(
    runtime_ids,
    root: str | None = None,
    *,
    min_criteria: int = 3,
    wide_metric_columns_min: int = 120,
):
    """Load exactly the hard cases whose opaque runtime_case_id is in ``runtime_ids``.

    This is the reproducible selector for a *frozen* preregistration: the runtime
    ids are label-safe hashes, so pinning them fixes the confirmatory case list
    without ever exposing ground-truth labels or label-derived paths.
    """
    wanted = {str(rid) for rid in runtime_ids}
    cases = load_hard_cases(
        root,
        sample=None,
        min_criteria=min_criteria,
        wide_metric_columns_min=wide_metric_columns_min,
    )
    by_runtime = {}
    for case in cases:
        rid = case.runtime_case_id() if hasattr(case, "runtime_case_id") else str(case.case_id)
        if rid in wanted:
            by_runtime[rid] = case
    ordered = [by_runtime[rid] for rid in runtime_ids if rid in by_runtime]
    missing = [rid for rid in runtime_ids if rid not in by_runtime]
    return ordered, missing


def write_hard_split(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build label-safe RCAEval hard split metadata")
    parser.add_argument("--root", default=None)
    parser.add_argument("--out", default="results/rcaeval_hard_split.json")
    parser.add_argument("--min-criteria", type=int, default=3)
    parser.add_argument("--wide-metric-columns-min", type=int, default=120)
    args = parser.parse_args(argv)
    payload = build_hard_split(
        args.root,
        min_criteria=args.min_criteria,
        wide_metric_columns_min=args.wide_metric_columns_min,
    )
    write_hard_split(args.out, payload)
    print(json.dumps({
        "out": args.out,
        "cases": len(payload["cases"]),
        "telemetry_median_bytes": payload["meta"]["telemetry_median_bytes"],
    }, indent=2))
    return 0


def _case_size(case_path: Path) -> int:
    if not case_path.exists():
        return 0
    total = 0
    for path in case_path.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _hard_row(
    root: Path,
    case_dir: Path,
    *,
    size_bytes: int,
    size_median: float,
    wide_metric_columns_min: int,
) -> dict[str, Any]:
    source = case_dir.parents[1].name
    suite, system = _source_parts(source)
    runtime_id = _runtime_case_id(root, case_dir, source)
    metric_columns = _metric_column_count(case_dir)
    criteria = {
        "suite_re2_or_re3": suite in {"RE2", "RE3"},
        "train_ticket_system": system == "TT",
        "telemetry_bytes_ge_median": size_bytes >= size_median,
        "has_logs": _has_logs(case_dir),
        "has_traces": _has_traces(case_dir),
        "wide_metric_table": metric_columns >= wide_metric_columns_min,
    }
    return {
        "runtime_case_id": runtime_id,
        "source": source,
        "suite": suite,
        "system": system,
        "telemetry_bytes": size_bytes,
        "telemetry_mb": round(size_bytes / 1_000_000, 4),
        "metric_columns": metric_columns,
        "criteria_score": sum(criteria.values()),
        "criteria": criteria,
    }


def _source_parts(source: str) -> tuple[str, str]:
    parts = source.split("-")
    suite = parts[0] if parts else source
    system = parts[1] if len(parts) > 1 else ""
    return suite, system


def _runtime_case_id(root: Path, case_dir: Path, source: str) -> str:
    internal_case_id = f"RCAEval-{source}-{case_dir.parent.name}-{case_dir.name}"
    digest = hashlib.sha256(internal_case_id.encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}"


def _has_logs(case_dir: Path) -> bool:
    return (case_dir / "logs.csv").exists() or (case_dir / "logts.csv").exists()


def _has_traces(case_dir: Path) -> bool:
    return (case_dir / "traces.csv").exists() or any(case_dir.glob("tracets_*.csv"))


def _metric_column_count(case_dir: Path) -> int:
    for name in ("simple_metrics.csv", "simple_data.csv", "data.csv", "metrics.csv"):
        path = case_dir / name
        if not path.exists():
            continue
        try:
            first_line = path.open(encoding="utf-8", errors="ignore").readline()
        except Exception:
            continue
        return max(0, len(first_line.rstrip("\n").split(",")) - 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

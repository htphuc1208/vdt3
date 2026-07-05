"""Lightweight RCAEval dataset adapter.

The adapter intentionally keeps ground-truth labels outside the inference
payload. Labels are inferred from RCAEval directory names only for scoring.
"""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .external import ExternalBenchmarkCase, ExternalPrediction
from .stats import accuracy_at_k, aggregate_ci, reciprocal_rank

EXPECTED_COUNTS = {"RE1": 375, "RE2": 270, "RE3": 90}
DEFAULT_RCAEVAL_ROOT = Path("data/rcaeval")


def resolve_rcaeval_root(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.getenv("RCAEVAL_DATA_DIR") or DEFAULT_RCAEVAL_ROOT).expanduser().resolve()


def validate_rcaeval(root: str | os.PathLike | None = None) -> dict:
    """Validate the expected RCAEval case counts."""
    root_path = resolve_rcaeval_root(root)
    counts = count_cases(root_path)
    total = sum(counts.values())
    expected_total = sum(EXPECTED_COUNTS.values())
    return {
        "root": str(root_path),
        "counts": counts,
        "total": total,
        "expected": EXPECTED_COUNTS,
        "expected_total": expected_total,
        "ok": counts == EXPECTED_COUNTS,
    }


def count_cases(root: str | os.PathLike | None = None) -> dict[str, int]:
    counts = {"RE1": 0, "RE2": 0, "RE3": 0}
    for case_dir in iter_case_dirs(root):
        suite = case_dir.parents[1].name.split("-")[0]
        if suite in counts:
            counts[suite] += 1
    return counts


def iter_case_dirs(root: str | os.PathLike | None = None) -> Iterable[Path]:
    root_path = resolve_rcaeval_root(root)
    if not root_path.exists():
        return []
    dirs: list[Path] = []
    for dataset_dir in sorted(root_path.glob("RE[123]-*")):
        if not dataset_dir.is_dir():
            continue
        for fault_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            for case_dir in sorted(p for p in fault_dir.iterdir() if p.is_dir()):
                if (
                    case_dir.name.isdigit()
                    and (case_dir / "inject_time.txt").is_file()
                    and _metric_path(case_dir) is not None
                ):
                    dirs.append(case_dir)
    return dirs


def load_cases(
    root: str | os.PathLike | None = None,
    *,
    sample: int | None = None,
    seed: int = 7,
) -> list[ExternalBenchmarkCase]:
    case_dirs = list(iter_case_dirs(root))
    if sample and sample > 0 and sample < len(case_dirs):
        rng = random.Random(seed)
        buckets: dict[str, list[Path]] = defaultdict(list)
        for case_dir in case_dirs:
            buckets[case_dir.parents[1].name].append(case_dir)
        selected_dirs: list[Path] = []
        per_bucket = max(1, sample // max(len(buckets), 1))
        for bucket in buckets.values():
            selected_dirs.extend(rng.sample(bucket, min(per_bucket, len(bucket))))
        remaining = [case_dir for case_dir in case_dirs if case_dir not in set(selected_dirs)]
        if len(selected_dirs) < sample:
            selected_dirs.extend(rng.sample(remaining, min(sample - len(selected_dirs), len(remaining))))
        cases = [_case_from_dir(case_dir) for case_dir in selected_dirs[:sample]]
        return sorted(cases, key=lambda c: c.case_id)
    return [_case_from_dir(case_dir) for case_dir in case_dirs]


def heuristic_predict(case: ExternalBenchmarkCase, system: str = "rcaeval_profile") -> ExternalPrediction:
    """Non-LLM telemetry-profile baseline for smoke runs and sanity checks."""
    ranked = [item["service"] for item in case.observability.get("top_metric_shifts", [])]
    root = ranked[0] if ranked else "UNKNOWN"
    fault_type = case.observability.get("likely_fault_family", "")
    return ExternalPrediction(
        case_id=case.case_id,
        system=system,
        root=root,
        ranked_roots=ranked,
        fault_type=str(fault_type),
        accepted=root != "UNKNOWN",
        confidence=float(case.observability.get("profile_confidence", 0.0)),
        notes="profile-delta heuristic over pre/post RCAEval telemetry",
    )


def score_predictions(cases: list[ExternalBenchmarkCase], predictions: list[ExternalPrediction]) -> dict:
    by_case = {case.case_id: case for case in cases}
    rows = []
    for pred in predictions:
        case = by_case[pred.case_id]
        ranked = pred.ranked_roots or [pred.root]
        row = {
            "case_id": pred.case_id,
            "source": case.source,
            "system": pred.system,
            "hit_at_1": accuracy_at_k(ranked, case.ground_truth_root, 1),
            "hit_at_3": accuracy_at_k(ranked, case.ground_truth_root, 3),
            "mrr": reciprocal_rank(ranked, case.ground_truth_root),
            "root_accuracy": accuracy_at_k([pred.root], case.ground_truth_root, 1),
            "fault_accuracy": _norm(pred.fault_type) == _norm(case.fault_type),
            "predicted_root": pred.root,
            "true_root": case.ground_truth_root,
            "predicted_fault_type": pred.fault_type,
            "true_fault_type": case.fault_type,
            "accepted": pred.accepted,
            "confidence": pred.confidence,
            "latency_s": pred.latency_s,
            "total_tokens": pred.total_tokens,
            "tool_calls": pred.tool_calls,
            "llm_calls": pred.llm_calls,
        }
        rows.append(row)
    summary = aggregate_ci(rows, ["hit_at_1", "hit_at_3", "mrr", "fault_accuracy", "accepted"])
    return {"summary": summary, "rows": rows}


def _case_from_dir(case_dir: Path) -> ExternalBenchmarkCase:
    dataset = case_dir.parents[1].name
    fault_label = case_dir.parent.name
    root, fault_type = _split_fault_label(fault_label)
    inject_time = _read_inject_time(case_dir)
    observability = _summarize_case(case_dir, root, fault_type, inject_time)
    instruction = (
        f"RCAEval {dataset} failure case with pre/post telemetry. "
        "Identify the root-cause service and fault indicator from the observed telemetry."
    )
    return ExternalBenchmarkCase(
        case_id=f"RCAEval-{dataset}-{fault_label}-{case_dir.name}",
        source=dataset,
        instruction=instruction,
        ground_truth_root=root,
        fault_type=fault_type,
        tags=dataset.split("-")[:2],
        observability=observability,
        label_extras={"case_path": str(case_dir), "inject_time": inject_time},
    )


def _summarize_case(case_dir: Path, root: str, fault_type: str, inject_time: int | None) -> dict:
    metric_path = _metric_path(case_dir)
    shifts = _top_metric_shifts(metric_path, inject_time) if metric_path else []
    top_score = shifts[0]["score"] if shifts else 0.0
    runner_up = shifts[1]["score"] if len(shifts) > 1 else 0.0
    confidence = top_score / (top_score + runner_up + 1e-9) if shifts else 0.0
    return {
        "metric_file": metric_path.name if metric_path else "",
        "inject_time": inject_time,
        "top_metric_shifts": shifts[:10],
        "likely_fault_family": _guess_fault_family(shifts, fault_type),
        "profile_confidence": round(confidence, 4),
        "has_logs": (case_dir / "logs.csv").exists() or (case_dir / "logts.csv").exists(),
        "has_traces": (case_dir / "traces.csv").exists() or any(case_dir.glob("tracets_*.csv")),
        "notes": "Labels are omitted from inference payload; root/fault are retained only for scoring.",
    }


def _top_metric_shifts(path: Path, inject_time: int | None, limit: int = 25) -> list[dict]:
    if path.suffix == ".json":
        try:
            rows = json.loads(path.read_text())
        except Exception:
            rows = []
    else:
        with path.open(newline="", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
    if not rows:
        return []

    columns = [c for c in rows[0] if c != "time" and not c.startswith("time.") and not c.endswith("latency-50")]
    if not columns:
        return []
    times = [_to_float(row.get("time")) for row in rows]
    if inject_time is None:
        split_idx = len(rows) // 2
        pre_rows, post_rows = rows[:split_idx], rows[split_idx:]
    else:
        pre_rows = [row for row, t in zip(rows, times) if t is not None and t < inject_time]
        post_rows = [row for row, t in zip(rows, times) if t is not None and t >= inject_time]
    if not pre_rows or not post_rows:
        split_idx = len(rows) // 2
        pre_rows, post_rows = rows[:split_idx], rows[split_idx:]

    items = []
    for column in columns:
        pre = [_to_float(row.get(column)) for row in pre_rows]
        post = [_to_float(row.get(column)) for row in post_rows]
        pre = [v for v in pre if v is not None]
        post = [v for v in post if v is not None]
        if not pre or not post:
            continue
        pre_mean = mean(pre)
        post_mean = mean(post)
        delta = post_mean - pre_mean
        score = abs(delta) / (abs(pre_mean) + 1.0)
        service, metric = split_metric_name(column)
        items.append({
            "service": service,
            "metric": metric,
            "column": column,
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "delta": round(delta, 4),
            "score": round(score, 6),
        })
    items.sort(key=lambda item: item["score"], reverse=True)
    dedup: dict[str, dict] = {}
    for item in items:
        dedup.setdefault(item["service"], item)
    return list(dedup.values())[:limit]


def split_metric_name(metric: str) -> tuple[str, str]:
    parts = metric.rsplit("_", 1)
    if len(parts) == 1:
        return metric, ""
    return parts[0], parts[1].replace("latency-90", "latency")


def _split_fault_label(label: str) -> tuple[str, str]:
    root, fault = label.rsplit("_", 1)
    return root, fault


def _metric_path(case_dir: Path) -> Path | None:
    for name in ("simple_metrics.csv", "simple_data.csv", "data.csv", "metrics.json", "metrics.csv"):
        path = case_dir / name
        if path.is_file():
            return path
    return None


def _read_inject_time(case_dir: Path) -> int | None:
    path = case_dir / "inject_time.txt"
    if not path.exists():
        return None
    try:
        return int(float(path.read_text().strip()))
    except Exception:
        return None


def _guess_fault_family(shifts: list[dict], fallback: str) -> str:
    if not shifts:
        return fallback
    metric = str(shifts[0].get("metric") or "").lower()
    for key in ("cpu", "mem", "disk", "delay", "loss", "socket"):
        if key in metric:
            return key
    return fallback


def _to_float(value) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _norm(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")

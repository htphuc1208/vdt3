"""Detailed OpenRCA error taxonomy for post-review debugging."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .tools import TELECOM_COMPONENTS, TELECOM_REASONS


def analyze_error_taxonomy(paths: list[str | Path]) -> dict[str, Any]:
    rows = _load_rows(paths)
    classified = [classify_error(row) for row in rows]
    category_counts: Counter[str] = Counter()
    for item in classified:
        category_counts.update(item["categories"])
    return {
        "benchmark": "openrca_telecom",
        "paths": [str(path) for path in paths],
        "row_count": len(classified),
        "systems": sorted({str(row.get("system")) for row in rows}),
        "category_counts": dict(sorted(category_counts.items())),
        "bias": _bias_tables(classified),
        "artifact_diagnostics": _artifact_diagnostics(rows),
        "rows": classified,
    }


def classify_error(row: dict[str, Any]) -> dict[str, Any]:
    expected = _expected(row.get("scoring_points", ""))
    parsed, parse_error = _parse_prediction(str(row.get("prediction") or ""))
    categories: list[str] = []
    if parse_error:
        categories.append("format")
    if expected["count"] and len(parsed) != expected["count"]:
        categories.append("multi_root_count")

    requested = {
        name
        for name, values in (
            ("component", expected["components"]),
            ("reason", expected["reasons"]),
            ("time", expected["times"]),
        )
        if values
    }
    missing = _missing_fields(parsed, requested)
    if missing:
        categories.append("missing_field")
    if _invalid_catalog(parsed):
        categories.append("invalid_catalog")
    if expected["components"] and not _components_match(parsed, expected["components"]):
        categories.append("component")
    if expected["reasons"] and not _reasons_match(parsed, expected["reasons"]):
        categories.append("reason")
    if expected["times"] and not _times_match(parsed, expected["times"]):
        categories.append("time")
    if bool(row.get("strict_correct")):
        categories = []
    elif not categories:
        categories.append("unknown")
    return {
        "row_id": str(row.get("row_id")),
        "task_index": str(row.get("task_index") or ""),
        "difficulty": _difficulty(str(row.get("task_index") or "")),
        "system": str(row.get("system") or ""),
        "strict_correct": bool(row.get("strict_correct")),
        "score": float(row.get("score") or 0.0),
        "categories": categories,
        "expected": expected,
        "predicted": parsed,
        "parse_error": parse_error,
        "predicted_component": parsed[0].get("root cause component", "") if parsed else "",
        "predicted_reason": parsed[0].get("root cause reason", "") if parsed else "",
        "volume_bin": str(row.get("volume_bin") or "unknown"),
        "artifact_summary": _row_artifact_summary(row),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify OpenRCA result errors by field and bias")
    parser.add_argument("paths", nargs="+", help="OpenRCA paired result JSON files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    payload = analyze_error_taxonomy(args.paths)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "row_count": payload["row_count"],
            "category_counts": payload["category_counts"],
        }, indent=2))
    else:
        print(text)
    return 0


def _load_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        default_system = payload.get("meta", {}).get("system")
        for row in payload.get("rows", []):
            item = dict(row)
            item["system"] = str(item.get("system") or default_system or Path(path).stem)
            rows.append(item)
    return rows


def _expected(scoring_points: str) -> dict[str, Any]:
    components = re.findall(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)", scoring_points)
    reasons = re.findall(r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)", scoring_points)
    times = re.findall(
        r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i\.e\., <=1min\) of ([^\n]+)",
        scoring_points,
    )
    return {
        "components": [item.strip() for item in components],
        "reasons": [item.strip() for item in reasons],
        "times": [item.strip() for item in times],
        "count": max(len(components), len(reasons), len(times)),
    }


def _parse_prediction(text: str) -> tuple[list[dict[str, str]], str]:
    try:
        payload = json.loads(text)
    except Exception as exc:
        return [], str(exc)
    if isinstance(payload, dict) and all(isinstance(value, dict) for value in payload.values()):
        items = [payload[key] for key in sorted(payload, key=lambda item: int(item) if str(item).isdigit() else str(item))]
    elif isinstance(payload, dict) and isinstance(payload.get("root_causes"), list):
        items = payload["root_causes"]
    elif isinstance(payload, list):
        items = payload
    else:
        return [], "prediction is not an OpenRCA object/list shape"
    parsed = []
    for raw in items:
        if not isinstance(raw, dict):
            return parsed, "prediction contains a non-object root cause"
        parsed.append({
            "root cause occurrence datetime": str(
                raw.get("root cause occurrence datetime")
                or raw.get("root_cause_occurrence_datetime")
                or ""
            ),
            "root cause component": str(
                raw.get("root cause component")
                or raw.get("root_cause_component")
                or ""
            ),
            "root cause reason": str(
                raw.get("root cause reason")
                or raw.get("root_cause_reason")
                or ""
            ),
        })
    return parsed, ""


def _missing_fields(parsed: list[dict[str, str]], requested: set[str]) -> list[str]:
    missing = []
    if not parsed:
        return sorted(requested)
    key_map = {
        "component": "root cause component",
        "reason": "root cause reason",
        "time": "root cause occurrence datetime",
    }
    for name in requested:
        key = key_map[name]
        if any(not item.get(key) for item in parsed):
            missing.append(name)
    return sorted(missing)


def _invalid_catalog(parsed: list[dict[str, str]]) -> bool:
    components = set(TELECOM_COMPONENTS)
    reasons = set(TELECOM_REASONS)
    for item in parsed:
        component = item.get("root cause component") or ""
        reason = item.get("root cause reason") or ""
        if component and component not in components:
            return True
        if reason and reason not in reasons:
            return True
    return False


def _components_match(parsed: list[dict[str, str]], expected: list[str]) -> bool:
    predicted = [item.get("root cause component") or "" for item in parsed]
    return sorted(predicted) == sorted(expected)


def _reasons_match(parsed: list[dict[str, str]], expected: list[str]) -> bool:
    predicted = [item.get("root cause reason") or "" for item in parsed]
    return sorted(predicted) == sorted(expected)


def _times_match(parsed: list[dict[str, str]], expected: list[str]) -> bool:
    predicted = [item.get("root cause occurrence datetime") or "" for item in parsed]
    if len(predicted) != len(expected):
        return False
    return all(_within_one_minute(exp, act) for exp, act in zip(sorted(expected), sorted(predicted)))


def _within_one_minute(expected: str, actual: str) -> bool:
    try:
        expected_dt = datetime.strptime(expected, "%Y-%m-%d %H:%M:%S")
        actual_dt = datetime.strptime(actual, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return abs(expected_dt - actual_dt).total_seconds() <= 60


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


def _bias_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables = {
        "predicted_component": Counter(),
        "predicted_reason": Counter(),
        "volume_bin": Counter(),
        "difficulty": Counter(),
    }
    by_system: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["strict_correct"]:
            continue
        tables["predicted_component"][row["predicted_component"] or "missing"] += 1
        tables["predicted_reason"][row["predicted_reason"] or "missing"] += 1
        tables["volume_bin"][row["volume_bin"]] += 1
        tables["difficulty"][row["difficulty"]] += 1
        by_system[row["system"]].update(row["categories"])
    return {
        **{name: dict(counter.most_common()) for name, counter in tables.items()},
        "by_system_category": {system: dict(counter.most_common()) for system, counter in by_system.items()},
    }


def _artifact_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    falsifier = Counter()
    fusion_winners = Counter()
    worker_tops = Counter()
    catalog_sources = Counter()
    for row in rows:
        artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
        fdiag = artifacts.get("falsifier") if isinstance(artifacts.get("falsifier"), dict) else {}
        if fdiag:
            falsifier[str(fdiag.get("selected") or "unknown")] += 1
        for candidate in artifacts.get("fusion_candidates", []) or []:
            if isinstance(candidate, dict):
                fusion_winners[str(candidate.get("component") or "missing")] += 1
                break
        for distribution in artifacts.get("worker_distributions", []) or []:
            if not isinstance(distribution, dict):
                continue
            candidates = distribution.get("candidates") or []
            if candidates and isinstance(candidates[0], dict):
                key = f"{distribution.get('worker_id')}:{candidates[0].get('component')}"
                worker_tops[key] += 1
        source = artifacts.get("candidate_catalog_source")
        if isinstance(source, dict):
            catalog_sources[json.dumps(source, sort_keys=True)] += 1
    return {
        "falsifier_selected": dict(falsifier.most_common()),
        "fusion_top_component": dict(fusion_winners.most_common()),
        "worker_top_component": dict(worker_tops.most_common()),
        "candidate_catalog_sources": {
            key: count for key, count in catalog_sources.most_common()
        },
    }


def _row_artifact_summary(row: dict[str, Any]) -> dict[str, Any]:
    artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
    pre = artifacts.get("pre_falsifier_winner") if isinstance(artifacts.get("pre_falsifier_winner"), dict) else {}
    falsifier = artifacts.get("falsifier") if isinstance(artifacts.get("falsifier"), dict) else {}
    return {
        "pre_falsifier_component": pre.get("component"),
        "pre_falsifier_reason": pre.get("reason"),
        "falsifier_selected": falsifier.get("selected"),
        "refinement_triggered": (artifacts.get("refinement") or {}).get("triggered")
        if isinstance(artifacts.get("refinement"), dict)
        else None,
        "fusion_candidate_count": len(artifacts.get("fusion_candidates") or []),
        "worker_distribution_count": len(artifacts.get("worker_distributions") or []),
    }


if __name__ == "__main__":
    raise SystemExit(main())

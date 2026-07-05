"""Training-only TelecomTS class catalog shared by MAS and single baselines."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from statistics import fmean, pstdev
from typing import Any

from .dataset import RCA_CLASSES, TelecomTSDataset
from .features import BASE_CLASS_CATALOG, PROTOTYPE_METRIC_KEYS, summarize_event


CATALOG_POLICY_ID = "development_affected_kpi_shape_mechanism_catalog_v3"


def build_training_catalog(
    dataset: TelecomTSDataset,
    *,
    split: str = "development",
) -> dict[str, dict[str, Any]]:
    if split != "development":
        raise ValueError("TelecomTS class catalog may only be learned from the development split")
    totals: Counter[str] = Counter()
    affected: dict[str, Counter[str]] = defaultdict(Counter)
    for record in dataset.rows(split):
        totals[record.root_cause] += 1
        values = record.raw["anomalies"].get("affected_kpis") or []
        affected[record.root_cause].update(str(value) for value in values)
    missing = [name for name in RCA_CLASSES if totals[name] == 0]
    if missing:
        raise ValueError(f"development split is missing catalog classes: {', '.join(missing)}")
    prototypes, event_support = _build_shape_prototypes(dataset, split=split)
    return {
        name: {
            **BASE_CLASS_CATALOG[name],
            "affected_kpis": [
                kpi for kpi, count in sorted(affected[name].items())
                if count == totals[name]
            ],
            "affected_kpi_count": sum(
                1 for count in affected[name].values() if count == totals[name]
            ),
            "training_support_windows": totals[name],
            "prototype_support_events": event_support[name],
            "shape_prototype": prototypes[name],
        }
        for name in RCA_CLASSES
    }


def catalog_sha256(catalog: dict[str, Any]) -> str:
    blob = json.dumps(catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _build_shape_prototypes(
    dataset: TelecomTSDataset,
    *,
    split: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    values: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    event_support: Counter[str] = Counter()
    for event_index, event in enumerate(dataset.events(split)):
        event_support[event.root_cause] += 1
        task = dataset.get_runtime_event(split, event_index)
        board = summarize_event(task["payload"])
        for shard_name, shard in board["shards"].items():
            for kpi_name, summary in shard.items():
                if summary.get("kind") != "numeric":
                    continue
                for metric in PROTOTYPE_METRIC_KEYS:
                    value = summary.get(metric)
                    if isinstance(value, int | float):
                        values[event.root_cause][shard_name][kpi_name][metric].append(float(value))
    prototypes: dict[str, dict[str, Any]] = {}
    for class_name in RCA_CLASSES:
        prototypes[class_name] = {
            shard_name: {
                kpi_name: {
                    metric: _mean_std(metric_values)
                    for metric, metric_values in sorted(metrics.items())
                }
                for kpi_name, metrics in sorted(kpis.items())
            }
            for shard_name, kpis in sorted(values[class_name].items())
        }
    return prototypes, {name: event_support.get(name, 0) for name in RCA_CLASSES}


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = fmean(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    return {"mean": _round(mean), "std": _round(std)}


def _round(value: float) -> float:
    return float(f"{value:.6g}")

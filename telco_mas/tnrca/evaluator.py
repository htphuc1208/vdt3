"""Set-based TN-RCA evaluation with per-case macro aggregation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .dataset import TNRCACase


def evaluate_predictions(
    cases: Sequence[TNRCACase],
    predictions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = {_normalize(value) for value in case.root_causes if _normalize(value)}
        predicted_values = predictions.get(case.case_id, ())
        predicted = {_normalize(value) for value in predicted_values if _normalize(value)}
        tp = len(expected & predicted)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(expected) if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "case_id": case.case_id,
                "expected": list(case.root_causes),
                "predicted": list(predicted_values),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "exact": predicted == expected,
            }
        )
    count = len(rows)
    return {
        "case_count": count,
        "macro_precision": sum(row["precision"] for row in rows) / count if count else 0.0,
        "macro_recall": sum(row["recall"] for row in rows) / count if count else 0.0,
        "macro_f1": sum(row["f1"] for row in rows) / count if count else 0.0,
        "exact_accuracy": sum(bool(row["exact"]) for row in rows) / count if count else 0.0,
        "rows": rows,
    }


def _normalize(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())

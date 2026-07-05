"""Conservative evaluator helpers for official TeleLogs result files."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .dataset import LABEL_KEY_NEEDLES, _normalise_key


@dataclass(frozen=True)
class TeleLogsScore:
    score_available: bool
    strict_correct: bool
    score: float
    method: str
    matched_labels: list[str]
    predicted_labels: list[str]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_available": self.score_available,
            "strict_correct": self.strict_correct,
            "score": self.score,
            "score_method": self.method,
            "matched_labels": self.matched_labels,
            "predicted_labels": self.predicted_labels,
            "score_reason": self.reason,
        }


def score_prediction(raw_row: dict[str, Any], prediction: dict[str, Any] | str) -> TeleLogsScore:
    truth = evaluator_label_set(raw_row)
    if not truth:
        return TeleLogsScore(False, False, 0.0, "telelogs_root_cause_set", [], [], "no evaluator labels found")
    predicted = prediction_label_set(prediction)
    if predicted:
        matched = sorted(truth & predicted)
        union = truth | predicted
        score = len(matched) / len(union) if union else 0.0
        return TeleLogsScore(
            True,
            predicted == truth,
            round(score, 4),
            "telelogs_root_cause_set",
            matched,
            sorted(predicted),
            "exact set match" if predicted == truth else "structured root cause set differs from evaluator labels",
        )

    pred_text = _normalise_text(_prediction_text(prediction))
    matched = sorted(label for label in truth if label and label in pred_text)
    score = len(matched) / len(truth)
    return TeleLogsScore(
        True,
        False,
        round(score, 4),
        "telelogs_label_text_recall",
        matched,
        [],
        "unstructured prediction; strict correctness requires a structured root_causes/root_cause field",
    )


def evaluator_label_set(row: dict[str, Any]) -> set[str]:
    values: list[str] = []
    _collect_label_values(row, values)
    return {_normalise_text(value) for value in values if len(_normalise_text(value)) >= 2}


def prediction_label_set(prediction: dict[str, Any] | str) -> set[str]:
    if not isinstance(prediction, dict):
        return set()
    values: list[str] = []
    for key in ("root_causes", "root_cause", "causes", "answer"):
        if key in prediction:
            _collect_values(prediction[key], values)
    return {_normalise_text(value) for value in values if len(_normalise_text(value)) >= 2}


def _collect_label_values(value: Any, out: list[str], *, key_is_label: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            is_label = key_is_label or _looks_label_key(str(key))
            _collect_label_values(child, out, key_is_label=is_label)
        return
    if isinstance(value, list):
        if key_is_label:
            _collect_values(value, out)
        else:
            for child in value:
                _collect_label_values(child, out, key_is_label=key_is_label)
        return
    if key_is_label and value is not None:
        out.append(str(value))


def _collect_values(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _collect_values(child, out)
        return
    if isinstance(value, list):
        for child in value:
            _collect_values(child, out)
        return
    if value is not None:
        out.append(str(value))


def _looks_label_key(key: str) -> bool:
    lower = _normalise_key(key)
    return any(needle in lower for needle in LABEL_KEY_NEEDLES)


def _prediction_text(prediction: dict[str, Any] | str) -> str:
    if isinstance(prediction, str):
        return prediction
    return " ".join(str(value) for value in prediction.values() if value is not None)


def _normalise_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

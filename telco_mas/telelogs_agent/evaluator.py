"""Evaluation helpers for local TeleLogsAgent result files.

This module intentionally separates runtime payloads from evaluator-only fields.
When official scoring code is unavailable, it uses conservative label-text
matching and marks the score method in every row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .dataset import LABEL_KEY_NEEDLES


@dataclass(frozen=True)
class TeleLogsScore:
    score_available: bool
    strict_correct: bool
    score: float
    method: str
    matched_label: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_available": self.score_available,
            "strict_correct": self.strict_correct,
            "score": self.score,
            "score_method": self.method,
            "matched_label": self.matched_label,
            "score_reason": self.reason,
        }


def score_prediction(raw_row: dict[str, Any], prediction: dict[str, Any] | str) -> TeleLogsScore:
    labels = evaluator_labels(raw_row)
    if not labels:
        return TeleLogsScore(
            score_available=False,
            strict_correct=False,
            score=0.0,
            method="label_text_match",
            reason="no evaluator label-like fields found in row",
        )
    pred_text = _normalise(_prediction_text(prediction))
    if not pred_text:
        return TeleLogsScore(
            score_available=True,
            strict_correct=False,
            score=0.0,
            method="label_text_match",
            reason="empty prediction",
        )
    best = ""
    for label in labels:
        norm = _normalise(label)
        if norm and norm in pred_text:
            if len(norm) > len(best):
                best = norm
    return TeleLogsScore(
        score_available=True,
        strict_correct=bool(best),
        score=1.0 if best else 0.0,
        method="label_text_match",
        matched_label=best,
        reason="matched evaluator label text" if best else "prediction did not contain evaluator label text",
    )


def evaluator_labels(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    _collect_label_values(row, values)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        norm = _normalise(text)
        if len(norm) < 3 or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _collect_label_values(value: Any, out: list[str], *, key_is_label: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            is_label = key_is_label or _looks_label_key(str(key))
            _collect_label_values(child, out, key_is_label=is_label)
        return
    if isinstance(value, list):
        for child in value:
            _collect_label_values(child, out, key_is_label=key_is_label)
        return
    if key_is_label and value is not None:
        out.append(str(value))


def _looks_label_key(key: str) -> bool:
    lower = key.lower().replace("-", "_")
    return any(needle in lower for needle in LABEL_KEY_NEEDLES)


def _prediction_text(prediction: dict[str, Any] | str) -> str:
    if isinstance(prediction, str):
        return prediction
    parts = []
    for key in ("root_cause", "answer", "diagnosis", "solution", "remediation", "rationale", "final_answer"):
        value = prediction.get(key)
        if value is not None:
            parts.append(str(value))
    if not parts:
        parts.append(str(prediction))
    return " ".join(parts)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

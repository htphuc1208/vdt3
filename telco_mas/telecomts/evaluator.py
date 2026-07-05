"""Strict closed-set scoring for TelecomTS root-cause predictions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .dataset import RCA_CLASSES


_CLASS_BY_NORMALIZED = {
    re.sub(r"[^a-z0-9]+", " ", name.lower()).strip(): name
    for name in RCA_CLASSES
}


@dataclass(frozen=True)
class TelecomTSScore:
    target: str
    predicted: str | None
    strict_correct: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "predicted": self.predicted,
            "strict_correct": self.strict_correct,
            "score": 1.0 if self.strict_correct else 0.0,
        }


def score_prediction(target: str, prediction: dict[str, Any]) -> TelecomTSScore:
    raw = prediction.get("root_cause")
    if raw is None:
        ranked = prediction.get("ranked_candidates")
        raw = ranked[0] if isinstance(ranked, list) and ranked else None
    predicted = canonical_class(raw)
    return TelecomTSScore(
        target=target,
        predicted=predicted,
        strict_correct=predicted == target,
    )


def canonical_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return _CLASS_BY_NORMALIZED.get(normalized)

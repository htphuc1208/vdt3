"""OpenRCA prediction formatting."""
from __future__ import annotations

import json
from typing import Any

from .schemas import OpenRCAPredictionOutput

FIELD_MAP = {
    "root_cause_occurrence_datetime": "root cause occurrence datetime",
    "root_cause_component": "root cause component",
    "root_cause_reason": "root cause reason",
}


def format_prediction(prediction: OpenRCAPredictionOutput | dict[str, Any]) -> str:
    if isinstance(prediction, dict):
        prediction = OpenRCAPredictionOutput.model_validate(prediction)
    payload: dict[str, dict[str, str]] = {}
    for index, item in enumerate(prediction.root_causes, start=1):
        raw = item.model_dump()
        out = {}
        for source, target in FIELD_MAP.items():
            value = raw.get(source)
            if value:
                out[target] = str(value)
        payload[str(index)] = out
    return json.dumps(payload, ensure_ascii=False)


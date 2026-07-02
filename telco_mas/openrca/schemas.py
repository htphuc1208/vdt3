"""Structured schemas for OpenRCA predictions."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

OpenRCAField = Literal[
    "root cause occurrence datetime",
    "root cause component",
    "root cause reason",
]


class OpenRCATaskParse(BaseModel):
    time_range_start: str | None = None
    time_range_end: str | None = None
    number_of_failures: int | None = Field(default=None, ge=1)
    requested_fields: list[OpenRCAField] = Field(default_factory=list)
    rationale: str = ""


class OpenRCAPredictionItem(BaseModel):
    root_cause_occurrence_datetime: str | None = None
    root_cause_component: str | None = None
    root_cause_reason: str | None = None


class OpenRCAPredictionOutput(BaseModel):
    root_causes: list[OpenRCAPredictionItem] = Field(min_length=1)
    rationale: str = ""


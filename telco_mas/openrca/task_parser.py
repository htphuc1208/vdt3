"""Deterministic, label-safe parsing of OpenRCA natural-language tasks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .schemas import OpenRCAField

OPENRCA_TIMEZONE = ZoneInfo("Asia/Shanghai")

TASK_FIELDS: dict[str, tuple[OpenRCAField, ...]] = {
    "task_1": ("root cause occurrence datetime",),
    "task_2": ("root cause reason",),
    "task_3": ("root cause component",),
    "task_4": ("root cause occurrence datetime", "root cause reason"),
    "task_5": ("root cause occurrence datetime", "root cause component"),
    "task_6": ("root cause component", "root cause reason"),
    "task_7": (
        "root cause occurrence datetime",
        "root cause component",
        "root cause reason",
    ),
}

_DATE_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{1,2}),\s+(\d{4})\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


@dataclass(frozen=True)
class ParsedOpenRCATask:
    row_id: int
    task_index: str
    instruction: str
    start: datetime
    end: datetime
    requested_fields: tuple[OpenRCAField, ...]
    number_of_failures: int

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1000)

    @property
    def end_ms(self) -> int:
        return int(self.end.timestamp() * 1000)

    @property
    def date_key(self) -> str:
        return self.start.strftime("%Y_%m_%d")

    def runtime_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "time_range_start": self.start.strftime("%Y-%m-%d %H:%M:%S"),
            "time_range_end": self.end.strftime("%Y-%m-%d %H:%M:%S"),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "timezone": "Asia/Shanghai",
            "requested_fields": list(self.requested_fields),
            "number_of_failures": self.number_of_failures,
            "date_key": self.date_key,
        }


def parse_runtime_task(task: dict[str, Any]) -> ParsedOpenRCATask:
    """Parse one runtime task without consulting scoring points or record.csv."""

    row_id = int(task["row_id"])
    task_index = str(task["task_index"])
    instruction = str(task["instruction"])
    requested_fields = TASK_FIELDS.get(task_index)
    if requested_fields is None:
        raise ValueError(f"Unsupported OpenRCA task_index: {task_index}")

    date_match = _DATE_RE.search(instruction)
    times = list(_TIME_RE.finditer(instruction))
    iso_match = _ISO_DATE_RE.search(instruction) if date_match is None else None
    if (date_match is None and iso_match is None) or len(times) < 2:
        raise ValueError(f"Could not parse OpenRCA time range for row_id={row_id}")

    if date_match is not None:
        month, day, year = date_match.groups()
        date = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
    else:
        assert iso_match is not None
        year, month, day = iso_match.groups()
        date = datetime(int(year), int(month), int(day))
    start = _with_time(date, times[0])
    end = _with_time(date, times[1])
    if end <= start:
        end += timedelta(days=1)

    return ParsedOpenRCATask(
        row_id=row_id,
        task_index=task_index,
        instruction=instruction,
        start=start,
        end=end,
        requested_fields=requested_fields,
        number_of_failures=_failure_count(instruction),
    )


def parse_all_runtime_tasks(dataset) -> list[ParsedOpenRCATask]:
    return [parse_runtime_task(dataset.get_runtime_task(row_id)) for row_id in range(len(dataset.rows))]


def _with_time(date: datetime, match: re.Match[str]) -> datetime:
    hour, minute, second = match.groups()
    return date.replace(
        hour=int(hour),
        minute=int(minute),
        second=int(second or 0),
        microsecond=0,
        tzinfo=OPENRCA_TIMEZONE,
    )


def _failure_count(instruction: str) -> int:
    lower = instruction.lower()
    if re.search(r"\b(single|one|1)\s+(?:system\s+)?failure\b", lower):
        return 1
    return 1

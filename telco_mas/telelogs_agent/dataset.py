"""Label-safe local loader for TeleLogsAgent benchmark files."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class TeleLogsAgentDatasetError(RuntimeError):
    pass


LABEL_KEY_NEEDLES = (
    "answer",
    "correct",
    "expected",
    "ground_truth",
    "label",
    "root_cause",
    "scoring",
    "solution",
    "target",
)


@dataclass(frozen=True)
class TeleLogsRuntimeTask:
    scenario_set: str
    row_id: int
    task_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_set": self.scenario_set,
            "row_id": self.row_id,
            "task_id": self.task_id,
            "payload": self.payload,
        }


class TeleLogsAgentDataset:
    """Loads TeleLogsAgent JSON files without exposing label-like fields."""

    DEFAULT_SETS = ("TS1", "TS2", "TS3")

    def __init__(self, root_dir: str | Path | None = None, *, scenario_sets: tuple[str, ...] | None = None) -> None:
        root = root_dir or os.getenv("TELELOGS_AGENT_DATA_DIR") or "data/telelogs_agent"
        self.root_dir = Path(root).expanduser().resolve()
        self.scenario_sets = tuple(scenario_sets or self.DEFAULT_SETS)
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._validate()

    def _validate(self) -> None:
        if not self.root_dir.exists():
            raise TeleLogsAgentDatasetError(f"TeleLogsAgent dataset directory does not exist: {self.root_dir}")
        missing = [name for name in self.scenario_sets if not self.test_path(name).exists()]
        if missing:
            raise TeleLogsAgentDatasetError(
                "Missing TeleLogsAgent test files: "
                + ", ".join(str(self.test_path(name)) for name in missing)
            )
        for name in self.scenario_sets:
            if not self.rows(name):
                raise TeleLogsAgentDatasetError(f"{self.test_path(name)} contains no tasks")

    def test_path(self, scenario_set: str) -> Path:
        return self.root_dir / scenario_set / "test.json"

    def rows(self, scenario_set: str) -> list[dict[str, Any]]:
        if scenario_set not in self._rows:
            self._rows[scenario_set] = _load_rows(self.test_path(scenario_set))
        return self._rows[scenario_set]

    def counts(self) -> dict[str, int]:
        return {name: len(self.rows(name)) for name in self.scenario_sets}

    def get_runtime_task(self, scenario_set: str, row_id: int) -> dict[str, Any]:
        rows = self.rows(scenario_set)
        if row_id < 0 or row_id >= len(rows):
            raise TeleLogsAgentDatasetError(f"{scenario_set} row_id {row_id} outside 0..{len(rows) - 1}")
        raw = rows[row_id]
        task_id = str(raw.get("id") or raw.get("task_id") or raw.get("scenario_id") or f"{scenario_set}-{row_id}")
        return TeleLogsRuntimeTask(
            scenario_set=scenario_set,
            row_id=row_id,
            task_id=task_id,
            payload=_strip_label_fields(raw),
        ).to_dict()

    def iter_runtime_tasks(self, selected: dict[str, list[int]] | None = None) -> Iterator[dict[str, Any]]:
        selected = selected or {name: list(range(len(self.rows(name)))) for name in self.scenario_sets}
        for name in self.scenario_sets:
            for row_id in selected.get(name, []):
                yield self.get_runtime_task(name, row_id)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = _first_list_value(data)
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _first_list_value(data: dict[str, Any]) -> list[Any]:
    for key in ("data", "items", "samples", "tasks", "test", "questions", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    for value in data.values():
        if isinstance(value, list):
            return value
    return []


def _strip_label_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_label_fields(item)
            for key, item in value.items()
            if not _looks_label_key(str(key))
        }
    if isinstance(value, list):
        return [_strip_label_fields(item) for item in value]
    return value


def _looks_label_key(key: str) -> bool:
    lower = key.lower().replace("-", "_")
    return any(needle in lower for needle in LABEL_KEY_NEEDLES)

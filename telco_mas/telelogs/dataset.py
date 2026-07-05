"""Label-safe loader for the official TeleLogs 5G RCA dataset."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class TeleLogsDatasetError(RuntimeError):
    pass


LABEL_KEY_NEEDLES = (
    "answer",
    "correct",
    "expected",
    "ground_truth",
    "label",
    "root_cause",
    "root_causes",
    "scoring",
    "solution",
    "target",
)


@dataclass(frozen=True)
class TeleLogsRuntimeTask:
    split: str
    row_id: int
    task_id: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "row_id": self.row_id,
            "task_id": self.task_id,
            "payload": self.payload,
        }


class TeleLogsDataset:
    """Loads local TeleLogs JSON files without exposing label-like fields."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        root = root_dir or os.getenv("TELELOGS_DATA_DIR") or "data/telelogs"
        self.root_dir = Path(root).expanduser().resolve()
        self._rows: dict[str, list[dict[str, Any]]] | None = None
        self._validate()

    def _validate(self) -> None:
        if not self.root_dir.exists():
            raise TeleLogsDatasetError(f"TeleLogs dataset directory does not exist: {self.root_dir}")
        if not self.json_paths():
            raise TeleLogsDatasetError(f"No TeleLogs JSON/JSONL files found under: {self.root_dir}")
        if not any(self.rows(split) for split in self.splits):
            raise TeleLogsDatasetError(f"No TeleLogs tasks found under: {self.root_dir}")

    def json_paths(self) -> list[Path]:
        out = []
        for suffix in ("*.json", "*.jsonl"):
            for path in sorted(self.root_dir.rglob(suffix)):
                if ".cache" in path.parts or path.name == "README.md":
                    continue
                out.append(path)
        return out

    @property
    def splits(self) -> tuple[str, ...]:
        rows = self._load_all_rows()
        ordered = [name for name in ("test", "validation", "valid", "dev", "train", "unknown") if name in rows]
        ordered.extend(sorted(name for name in rows if name not in ordered))
        return tuple(ordered)

    def rows(self, split: str = "test") -> list[dict[str, Any]]:
        rows = self._load_all_rows()
        if split not in rows:
            raise TeleLogsDatasetError(
                f"TeleLogs split '{split}' not found. Available splits: {', '.join(self.splits) or 'none'}"
            )
        return rows[split]

    def counts(self) -> dict[str, int]:
        return {split: len(self.rows(split)) for split in self.splits}

    def get_runtime_task(self, split: str, row_id: int) -> dict[str, Any]:
        rows = self.rows(split)
        if row_id < 0 or row_id >= len(rows):
            raise TeleLogsDatasetError(f"{split} row_id {row_id} outside 0..{len(rows) - 1}")
        raw = rows[row_id]
        task_id = str(raw.get("id") or raw.get("task_id") or raw.get("sample_id") or f"{split}-{row_id}")
        return TeleLogsRuntimeTask(
            split=split,
            row_id=row_id,
            task_id=task_id,
            payload=_strip_label_fields(raw),
        ).to_dict()

    def iter_runtime_tasks(self, split: str, row_ids: list[int] | None = None) -> Iterator[dict[str, Any]]:
        ids = row_ids if row_ids is not None else list(range(len(self.rows(split))))
        for row_id in ids:
            yield self.get_runtime_task(split, row_id)

    def _load_all_rows(self) -> dict[str, list[dict[str, Any]]]:
        if self._rows is not None:
            return self._rows
        out: dict[str, list[dict[str, Any]]] = {}
        for path in self.json_paths():
            path_split = _split_from_path(path)
            for split, row in _load_rows_from_file(path, default_split=path_split):
                out.setdefault(split, []).append(row)
        self._rows = out
        return out


def _load_rows_from_file(path: Path, *, default_split: str) -> list[tuple[str, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append((default_split, value))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    return _extract_rows(data, default_split=default_split)


def _extract_rows(value: Any, *, default_split: str) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, list):
        return [(default_split, item) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, child in value.items():
        split = _normalise_split(str(key)) or default_split
        if isinstance(child, list) and all(isinstance(item, dict) for item in child):
            out.extend((split, item) for item in child)
        elif isinstance(child, dict):
            out.extend(_extract_rows(child, default_split=split))
    if out:
        return out
    if _looks_like_task_row(value):
        return [(default_split, value)]
    return []


def _looks_like_task_row(row: dict[str, Any]) -> bool:
    keys = {_normalise_key(key) for key in row}
    return bool(keys & {"question", "prompt", "context", "symptom", "root_cause", "root_causes", "answer"})


def _split_from_path(path: Path) -> str:
    tokens = [_normalise_split(part) for part in [*path.parts, path.stem]]
    for preferred in ("test", "validation", "valid", "dev", "train"):
        if preferred in tokens:
            return preferred
    return "unknown"


def _normalise_split(value: str) -> str:
    lower = value.lower()
    if lower in {"test", "tests"}:
        return "test"
    if lower in {"train", "training"}:
        return "train"
    if lower in {"validation", "valid", "val"}:
        return "validation"
    if lower in {"dev", "development"}:
        return "dev"
    return ""


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
    lower = _normalise_key(key)
    return any(needle in lower for needle in LABEL_KEY_NEEDLES)


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")

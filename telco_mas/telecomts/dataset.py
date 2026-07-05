"""Label-safe loader for the TelecomTS root-cause classification task."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator


SOURCE_URL = "https://huggingface.co/datasets/AliMaatouk/TelecomTS"
SPLIT_POLICY_ID = "latin_square_source_session_v1"
EXPECTED_SOURCE_SESSIONS = {
    f"{zone}/{application}"
    for zone in ("Zone_A", "Zone_B", "Zone_C")
    for application in ("File", "Twitch", "YouTube")
}
RCA_CLASSES = (
    "Antenna Failure",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Buffer Overflow (Gradual Buildup)",
    "Resource Allocation Bugs",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
)
KPI_NAMES = (
    "RSRP",
    "DL_BLER",
    "DL_MCS",
    "UL_BLER",
    "UL_MCS",
    "UL_NPRB",
    "UL_SNR",
    "TX_Bytes",
    "RX_Bytes",
    "Estimated_UL_Buffer",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "UL_Protocol",
    "UL_NumberOfPackets",
    "DL_Protocol",
    "DL_NumberOfPackets",
)

_ZONE_INDEX = {"Zone_A": 0, "Zone_B": 1, "Zone_C": 2}
_APPLICATION_INDEX = {"File": 0, "Twitch": 1, "YouTube": 2}
_FOLD_TO_SPLIT = {0: "development", 1: "validation", 2: "test"}


class TelecomTSDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelecomTSRecord:
    source_path: str
    source_line: int
    source_session: str
    split: str
    case_id: str
    raw: dict[str, Any]

    @property
    def root_cause(self) -> str:
        return str(self.raw["anomalies"]["type"])


@dataclass(frozen=True)
class TelecomTSEvent:
    source_session: str
    split: str
    event_id: str
    root_cause: str
    event_start: datetime
    event_end: datetime
    records: tuple[TelecomTSRecord, ...]


class TelecomTSDataset:
    """Loads the ten-class upstream RCA task without exposing evaluator labels."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        root = root_dir or os.getenv("TELECOMTS_DATA_DIR") or "data/telecomts"
        self.root_dir = Path(root).expanduser().resolve()
        self._records: list[TelecomTSRecord] | None = None
        self._events: list[TelecomTSEvent] | None = None
        self._validate_root()

    @property
    def splits(self) -> tuple[str, ...]:
        return ("development", "validation", "test")

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(sorted({record.source_session for record in self.records()}))

    @property
    def complete_official_layout(self) -> bool:
        return set(self.source_sessions) == EXPECTED_SOURCE_SESSIONS

    def jsonl_paths(self) -> list[Path]:
        return sorted(self.root_dir.glob("anomalous/synthetic/*/*/processed/chunked.jsonl"))

    def records(self) -> list[TelecomTSRecord]:
        if self._records is None:
            self._records = self._load_records()
        return self._records

    def rows(self, split: str) -> list[TelecomTSRecord]:
        if split not in self.splits:
            raise TelecomTSDatasetError(
                f"Unknown TelecomTS split '{split}'; expected one of {', '.join(self.splits)}"
            )
        return [record for record in self.records() if record.split == split]

    def counts(self) -> dict[str, int]:
        return {split: len(self.rows(split)) for split in self.splits}

    def class_counts(self, split: str | None = None) -> dict[str, int]:
        records = self.rows(split) if split else self.records()
        counts = Counter(record.root_cause for record in records)
        return {name: counts.get(name, 0) for name in RCA_CLASSES}

    def events(self, split: str | None = None) -> list[TelecomTSEvent]:
        if split is not None and split not in self.splits:
            raise TelecomTSDatasetError(
                f"Unknown TelecomTS split '{split}'; expected one of {', '.join(self.splits)}"
            )
        if self._events is None:
            self._events = self._build_events()
        if split is None:
            return self._events
        return [event for event in self._events if event.split == split]

    def event_counts(self) -> dict[str, int]:
        return {split: len(self.events(split)) for split in self.splits}

    def class_event_counts(self, split: str | None = None) -> dict[str, int]:
        counts = Counter(event.root_cause for event in self.events(split))
        return {name: counts.get(name, 0) for name in RCA_CLASSES}

    def get_record(self, split: str, row_id: int) -> TelecomTSRecord:
        rows = self.rows(split)
        if row_id < 0 or row_id >= len(rows):
            raise TelecomTSDatasetError(f"{split} row_id {row_id} outside 0..{len(rows) - 1}")
        return rows[row_id]

    def get_runtime_task(self, split: str, row_id: int) -> dict[str, Any]:
        record = self.get_record(split, row_id)
        raw = record.raw
        labels = raw["labels"]
        return {
            "benchmark": "telecomts",
            "split": split,
            "row_id": row_id,
            "case_id": record.case_id,
            "payload": {
                "sampling_rate_hz": raw["sampling_rate"],
                "sample_length": 128,
                "scenario": {
                    "zone": labels["zone"],
                    "application": labels["application"],
                    "mobility": labels["mobility"],
                    "congestion": labels["congestion"],
                },
                "kpis": raw["KPIs"],
                "candidate_root_causes": list(RCA_CLASSES),
            },
        }

    def iter_runtime_tasks(
        self,
        split: str,
        row_ids: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        ids = row_ids if row_ids is not None else list(range(len(self.rows(split))))
        for row_id in ids:
            yield self.get_runtime_task(split, row_id)

    def target(self, split: str, row_id: int) -> str:
        return self.get_record(split, row_id).root_cause

    def get_runtime_event(self, split: str, event_index: int) -> dict[str, Any]:
        events = self.events(split)
        if event_index < 0 or event_index >= len(events):
            raise TelecomTSDatasetError(
                f"{split} event_index {event_index} outside 0..{len(events) - 1}"
            )
        event = events[event_index]
        kpis = _merge_event_kpis(event)
        labels = event.records[0].raw["labels"]
        return {
            "benchmark": "telecomts",
            "split": split,
            "event_index": event_index,
            "case_id": event.event_id,
            "payload": {
                "sampling_rate_hz": 10,
                "sample_length": len(next(iter(kpis.values()))),
                "scenario": {
                    "zone": labels["zone"],
                    "application": labels["application"],
                    "mobility": labels["mobility"],
                    "congestion": labels["congestion"],
                },
                "kpis": kpis,
                "candidate_root_causes": list(RCA_CLASSES),
            },
        }

    def iter_runtime_events(
        self,
        split: str,
        event_indices: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        indices = event_indices if event_indices is not None else list(range(len(self.events(split))))
        for event_index in indices:
            yield self.get_runtime_event(split, event_index)

    def event_target(self, split: str, event_index: int) -> str:
        events = self.events(split)
        if event_index < 0 or event_index >= len(events):
            raise TelecomTSDatasetError(
                f"{split} event_index {event_index} outside 0..{len(events) - 1}"
            )
        return events[event_index].root_cause

    def _validate_root(self) -> None:
        if not self.root_dir.exists():
            raise TelecomTSDatasetError(f"TelecomTS dataset directory does not exist: {self.root_dir}")
        if not self.jsonl_paths():
            raise TelecomTSDatasetError(
                "No TelecomTS synthetic RCA JSONL files found under "
                f"{self.root_dir}/anomalous/synthetic"
            )

    def _load_records(self) -> list[TelecomTSRecord]:
        records: list[TelecomTSRecord] = []
        for path in self.jsonl_paths():
            relative = path.relative_to(self.root_dir).as_posix()
            source_session = _source_session(path)
            split = _split_for_session(source_session)
            with path.open(encoding="utf-8") as handle:
                for source_line, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TelecomTSDatasetError(
                            f"Invalid JSON in {relative}:{source_line}: {exc}"
                        ) from exc
                    _validate_row(raw, relative=relative, source_line=source_line)
                    records.append(TelecomTSRecord(
                        source_path=relative,
                        source_line=source_line,
                        source_session=source_session,
                        split=split,
                        case_id=_opaque_case_id(relative, source_line),
                        raw=raw,
                    ))
        if not records:
            raise TelecomTSDatasetError(f"No TelecomTS RCA rows found under: {self.root_dir}")
        return records

    def _build_events(self) -> list[TelecomTSEvent]:
        grouped: dict[tuple[str, str], list[tuple[datetime, datetime, TelecomTSRecord]]] = {}
        for record in self.records():
            start, end = _anomaly_interval(record)
            grouped.setdefault((record.source_session, record.root_cause), []).append(
                (start, end, record)
            )

        events: list[TelecomTSEvent] = []
        tolerance = timedelta(seconds=0.11)
        for (source_session, root_cause), intervals in sorted(grouped.items()):
            clusters: list[list[Any]] = []
            for start, end, record in sorted(intervals, key=lambda item: (item[0], item[1])):
                if not clusters or start > clusters[-1][1] + tolerance:
                    clusters.append([start, end, [record]])
                else:
                    clusters[-1][1] = max(clusters[-1][1], end)
                    clusters[-1][2].append(record)
            for start, end, records in clusters:
                split = records[0].split
                events.append(TelecomTSEvent(
                    source_session=source_session,
                    split=split,
                    event_id=_opaque_event_id(source_session, start, end),
                    root_cause=root_cause,
                    event_start=start,
                    event_end=end,
                    records=tuple(sorted(records, key=lambda item: item.raw["start_time"])),
                ))
        return sorted(events, key=lambda event: (event.split, event.source_session, event.event_start))


def _source_session(path: Path) -> str:
    parts = path.parts
    try:
        index = parts.index("synthetic")
        zone, application = parts[index + 1], parts[index + 2]
    except (ValueError, IndexError) as exc:
        raise TelecomTSDatasetError(f"Unexpected TelecomTS path: {path}") from exc
    if zone not in _ZONE_INDEX or application not in _APPLICATION_INDEX:
        raise TelecomTSDatasetError(f"Unknown TelecomTS source session in path: {path}")
    return f"{zone}/{application}"


def _split_for_session(source_session: str) -> str:
    zone, application = source_session.split("/", maxsplit=1)
    fold = (_APPLICATION_INDEX[application] - _ZONE_INDEX[zone]) % 3
    return _FOLD_TO_SPLIT[fold]


def _validate_row(raw: Any, *, relative: str, source_line: int) -> None:
    where = f"{relative}:{source_line}"
    if not isinstance(raw, dict):
        raise TelecomTSDatasetError(f"TelecomTS row is not an object at {where}")
    anomaly = raw.get("anomalies")
    if not isinstance(anomaly, dict) or anomaly.get("exists") is not True:
        raise TelecomTSDatasetError(f"TelecomTS RCA row is not anomalous at {where}")
    root_cause = anomaly.get("type")
    if root_cause not in RCA_CLASSES:
        raise TelecomTSDatasetError(f"Unsupported TelecomTS RCA class {root_cause!r} at {where}")
    kpis = raw.get("KPIs")
    if not isinstance(kpis, dict) or set(kpis) != set(KPI_NAMES):
        raise TelecomTSDatasetError(f"Unexpected TelecomTS KPI schema at {where}")
    for name in KPI_NAMES:
        values = kpis[name]
        if not isinstance(values, list) or len(values) != 128:
            raise TelecomTSDatasetError(f"KPI {name} must contain 128 values at {where}")
    labels = raw.get("labels")
    required_labels = {"zone", "application", "mobility", "congestion", "anomaly_present"}
    if not isinstance(labels, dict) or not required_labels.issubset(labels):
        raise TelecomTSDatasetError(f"Missing TelecomTS scenario labels at {where}")
    if raw.get("sampling_rate") != 10:
        raise TelecomTSDatasetError(f"Unexpected TelecomTS sampling rate at {where}")


def _opaque_case_id(relative: str, source_line: int) -> str:
    digest = hashlib.sha256(f"{relative}:{source_line}".encode("utf-8")).hexdigest()[:16]
    return f"TTS-{digest}"


def _anomaly_interval(record: TelecomTSRecord) -> tuple[datetime, datetime]:
    raw = record.raw
    sample_start = datetime.fromisoformat(str(raw["start_time"]))
    duration = raw["anomalies"]["anomaly_duration"]
    rate = float(raw["sampling_rate"])
    return (
        sample_start + timedelta(seconds=float(duration["start"]) / rate),
        sample_start + timedelta(seconds=float(duration["end"]) / rate),
    )


def _opaque_event_id(source_session: str, start: datetime, end: datetime) -> str:
    value = f"{source_session}:{start.isoformat()}:{end.isoformat()}"
    return f"TTE-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _merge_event_kpis(event: TelecomTSEvent) -> dict[str, list[Any]]:
    sample_starts = [datetime.fromisoformat(str(record.raw["start_time"])) for record in event.records]
    origin = min(sample_starts)
    merged: dict[str, dict[int, Any]] = {name: {} for name in KPI_NAMES}
    for record, sample_start in zip(event.records, sample_starts):
        offset = round((sample_start - origin).total_seconds() * 10)
        for name in KPI_NAMES:
            for local_index, value in enumerate(record.raw["KPIs"][name]):
                index = offset + local_index
                if index in merged[name] and merged[name][index] != value:
                    raise TelecomTSDatasetError(
                        f"Conflicting overlap for {event.event_id} KPI {name} at index {index}"
                    )
                merged[name][index] = value
    expected = set(range(max(max(values) for values in merged.values()) + 1))
    output: dict[str, list[Any]] = {}
    for name, values in merged.items():
        if set(values) != expected:
            raise TelecomTSDatasetError(f"Non-contiguous merged KPI {name} for {event.event_id}")
        output[name] = [values[index] for index in sorted(values)]
    return output

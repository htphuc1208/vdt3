"""Telemetry cataloging and windowed extraction for ShardRCA."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


class TelemetryFileInfo(BaseModel):
    path: str
    relative_path: str
    modality: str
    size_bytes: int = 0
    row_count: int = 0
    timestamp_column: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    columns_preview: list[str] = Field(default_factory=list)


class TelemetryCatalog(BaseModel):
    dataset: str
    case_id: str
    root: str
    query_time: float | None = None
    time_range_start: float | None = None
    time_range_end: float | None = None
    files: list[TelemetryFileInfo] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def files_by_modality(self, modality: str) -> list[TelemetryFileInfo]:
        return [item for item in self.files if item.modality == modality]

    def summary(self) -> dict[str, Any]:
        by_modality: dict[str, dict[str, Any]] = {}
        for item in self.files:
            bucket = by_modality.setdefault(item.modality, {"files": 0, "bytes": 0, "rows": 0})
            bucket["files"] += 1
            bucket["bytes"] += item.size_bytes
            bucket["rows"] += item.row_count
        return {
            "dataset": self.dataset,
            "case_id": self.case_id,
            "query_time": self.query_time,
            "time_range": [self.time_range_start, self.time_range_end],
            "modalities": by_modality,
            "file_count": len(self.files),
            "total_bytes": sum(item.size_bytes for item in self.files),
        }


class ShardSpec(BaseModel):
    shard_id: str
    modality: str
    paths: list[str]
    query_time: float | None = None
    start_time: float | None = None
    end_time: float | None = None
    components: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_catalog(
    root: str | os.PathLike,
    *,
    dataset: str,
    case_id: str,
    query_time: float | int | str | None = None,
    time_range_start: float | int | str | None = None,
    time_range_end: float | int | str | None = None,
    compute_ranges: bool = True,
    max_files: int | None = None,
) -> TelemetryCatalog:
    """Build a file catalog without loading full telemetry into memory."""

    root_path = Path(root).expanduser().resolve()
    files: list[TelemetryFileInfo] = []
    candidates = sorted(p for p in root_path.rglob("*") if p.is_file() and _supported_file(p))
    if max_files is not None:
        candidates = candidates[: max(0, max_files)]
    for path in candidates:
        files.append(_file_info(path, root_path, compute_ranges=compute_ranges))
    return TelemetryCatalog(
        dataset=dataset,
        case_id=case_id,
        root=str(root_path),
        query_time=_parse_time_value(query_time),
        time_range_start=_parse_time_value(time_range_start),
        time_range_end=_parse_time_value(time_range_end),
        files=files,
    )


def build_catalog_for_case(case, *, compute_ranges: bool = True) -> TelemetryCatalog:
    """Create a catalog for an ExternalBenchmarkCase without exposing labels."""

    case_path = Path(case.label_extras.get("case_path", "")).expanduser().resolve()
    query_time = case.label_extras.get("inject_time") or case.observability.get("inject_time")
    runtime_case_id = case.runtime_case_id() if hasattr(case, "runtime_case_id") else str(case.case_id)
    return build_catalog(
        case_path,
        dataset=str(case.source),
        case_id=runtime_case_id,
        query_time=query_time,
        compute_ranges=compute_ranges,
    )


def make_default_shards(
    catalog: TelemetryCatalog,
    *,
    max_shards: int = 3,
    prefer_simple_metrics: bool = True,
) -> list[ShardSpec]:
    """Plan simple disjoint modality shards from a telemetry catalog."""

    shards: list[ShardSpec] = []
    for modality in ("metrics", "logs", "traces"):
        files = catalog.files_by_modality(modality)
        if not files:
            continue
        selected = files
        if modality == "metrics" and prefer_simple_metrics:
            simple = [
                item for item in files
                if Path(item.path).name in {"simple_metrics.csv", "simple_data.csv", "data.csv", "metrics.json"}
            ]
            if simple:
                selected = simple
        shards.append(
            ShardSpec(
                shard_id=f"{modality}_0",
                modality=modality,
                paths=[item.path for item in selected],
                query_time=catalog.query_time,
                start_time=catalog.time_range_start,
                end_time=catalog.time_range_end,
                metadata={
                    "file_count": len(selected),
                    "bytes": sum(item.size_bytes for item in selected),
                },
            )
        )
    if len(shards) <= max_shards:
        return shards
    return shards[:max_shards]


def make_component_group_shards(
    catalog: TelemetryCatalog,
    *,
    group_size: int = 6,
    max_groups: int | None = None,
) -> list[ShardSpec]:
    """Create modality shards over disjoint component groups.

    This is the key v5 split for RCAEval-style wide telemetry. A single agent
    cannot inspect every component group under a small tool budget, while MAS can
    mine groups in parallel.
    """

    components = _candidate_components(catalog)
    if not components:
        return make_default_shards(catalog, max_shards=3)
    groups = [components[i : i + group_size] for i in range(0, len(components), group_size)]
    if max_groups is not None:
        groups = groups[: max(0, max_groups)]
    shards: list[ShardSpec] = []
    for group_idx, group in enumerate(groups):
        for modality in ("metrics", "logs", "traces"):
            files = catalog.files_by_modality(modality)
            if not files:
                continue
            shards.append(
                ShardSpec(
                    shard_id=f"{modality}_g{group_idx}",
                    modality=modality,
                    paths=[item.path for item in files],
                    query_time=catalog.query_time,
                    start_time=catalog.time_range_start,
                    end_time=catalog.time_range_end,
                    components=group,
                    metadata={
                        "component_group": group,
                        "group_index": group_idx,
                        "file_count": len(files),
                        "bytes": sum(item.size_bytes for item in files),
                    },
                )
            )
    return shards or make_default_shards(catalog, max_shards=3)


def candidate_components(catalog: TelemetryCatalog) -> list[str]:
    """Return the label-safe component universe inferred from metric schema."""

    return _candidate_components(catalog)


def extract_window_csv(
    source_path: str | os.PathLike,
    out_path: str | os.PathLike,
    *,
    start_time: float | int | str,
    end_time: float | int | str,
    timestamp_column: str | None = None,
    chunksize: int = 50_000,
) -> dict[str, Any]:
    """Write a time-windowed CSV extract using chunked pandas IO."""

    pd = _pd()
    source = Path(source_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = pd.read_csv(source, nrows=0)
    time_col = timestamp_column or _choose_timestamp_column(list(header.columns))
    if time_col is None:
        raise ValueError(f"No timestamp column found in {source}")
    start = _parse_time_value(start_time)
    end = _parse_time_value(end_time)
    if start is None or end is None:
        raise ValueError("start_time and end_time must be parseable")

    wrote_header = False
    rows_written = 0
    chunks = 0
    for chunk in pd.read_csv(source, chunksize=chunksize):
        chunks += 1
        times = _coerce_time_series(chunk[time_col])
        aligned_start, aligned_end = _align_bounds_to_series(start, end, times)
        mask = times.between(aligned_start, aligned_end, inclusive="both")
        filtered = chunk.loc[mask]
        if filtered.empty:
            continue
        filtered.to_csv(out, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        rows_written += int(len(filtered))
    if not wrote_header:
        header.to_csv(out, index=False)
    return {
        "source": str(source),
        "out": str(out),
        "timestamp_column": time_col,
        "chunks": chunks,
        "rows_written": rows_written,
    }


def _file_info(path: Path, root: Path, *, compute_ranges: bool) -> TelemetryFileInfo:
    columns: list[str] = []
    timestamp_column = None
    row_count = 0
    start_time = None
    end_time = None
    if path.suffix.lower() == ".csv":
        try:
            pd = _pd()
            header = pd.read_csv(path, nrows=0)
            columns = [str(col) for col in header.columns]
            timestamp_column = _choose_timestamp_column(columns)
            if compute_ranges and timestamp_column:
                row_count, start_time, end_time = _scan_time_range(path, timestamp_column)
        except Exception:
            columns = []
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            columns = list(parquet.schema.names)
            timestamp_column = _choose_timestamp_column(columns)
            row_count = int(parquet.metadata.num_rows)
            if compute_ranges and timestamp_column and row_count:
                pd = _pd()
                frame = pd.read_parquet(path, columns=[timestamp_column])
                values = _coerce_time_series(frame[timestamp_column]).dropna()
                if not values.empty:
                    start_time = float(values.min())
                    end_time = float(values.max())
        except Exception:
            columns = []
    return TelemetryFileInfo(
        path=str(path),
        relative_path=str(path.relative_to(root)) if path.is_relative_to(root) else path.name,
        modality=_infer_modality(path),
        size_bytes=path.stat().st_size,
        row_count=row_count,
        timestamp_column=timestamp_column,
        start_time=start_time,
        end_time=end_time,
        columns_preview=columns[:20],
    )


def _candidate_components(catalog: TelemetryCatalog) -> list[str]:
    components: set[str] = set()
    for item in catalog.files_by_modality("metrics"):
        path = Path(item.path)
        if path.suffix.lower() != ".csv":
            continue
        try:
            header = path.open(encoding="utf-8", errors="ignore").readline().strip()
        except Exception:
            continue
        for column in header.split(","):
            if column.lower() in {"time", "timestamp", "datetime", "date"}:
                continue
            component = _component_from_metric_column(column)
            if component and _valid_candidate_component(catalog, component):
                components.add(component)
    return sorted(components)


def _component_from_metric_column(column: str) -> str:
    if "_" in column:
        return column.rsplit("_", 1)[0]
    for marker in ("_container-", "_istio-", "_node-"):
        if marker in column:
            return column.split(marker, 1)[0]
    return ""


def _valid_candidate_component(catalog: TelemetryCatalog, component: str) -> bool:
    lower = component.lower()
    if str(catalog.dataset).startswith("RE"):
        if lower.startswith(("ip-", "gke-", "node-", "loadgenerator")):
            return False
        if "compute.internal" in lower:
            return False
    return bool(component)


def _scan_time_range(path: Path, timestamp_column: str, *, chunksize: int = 50_000) -> tuple[int, float | None, float | None]:
    pd = _pd()
    row_count = 0
    start_time = None
    end_time = None
    for chunk in pd.read_csv(path, usecols=[timestamp_column], chunksize=chunksize):
        row_count += int(len(chunk))
        values = _coerce_time_series(chunk[timestamp_column]).dropna()
        if values.empty:
            continue
        current_min = float(values.min())
        current_max = float(values.max())
        start_time = current_min if start_time is None else min(start_time, current_min)
        end_time = current_max if end_time is None else max(end_time, current_max)
    return row_count, start_time, end_time


def _supported_file(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".parquet", ".json", ".log", ".txt"}


def _infer_modality(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "metric" in name or "metric" in parent or name in {"data.csv", "simple_data.csv"}:
        return "metrics"
    if "trace" in name or "trace" in parent or "span" in name:
        return "traces"
    if "log" in name or "log" in parent:
        return "logs"
    if name.endswith(".json"):
        return "auxiliary"
    return "events"


def _choose_timestamp_column(columns: Iterable[str]) -> str | None:
    names = list(columns)
    lower = {name.lower(): name for name in names}
    for candidate in (
        "timestamp",
        "starttimemillis",
        "start_time_millis",
        "start_time",
        "endtimemillis",
        "ts",
        "time",
        "datetime",
        "date",
        "starttime",
    ):
        if candidate in lower:
            return lower[candidate]
    for name in names:
        compact = re.sub(r"[^a-z0-9]", "", name.lower())
        if compact in {"timestamp", "starttimemillis", "time", "datetime"}:
            return name
    return None


def _parse_time_value(value: float | int | str | None) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        pass
    try:
        pd = _pd()
        parsed = pd.to_datetime(value, errors="coerce")
        if parsed is None or pd.isna(parsed):
            return None
        return float(parsed.timestamp())
    except Exception:
        return None


def _coerce_time_series(series):
    pd = _pd()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype("float64")
    parsed = pd.to_datetime(series, errors="coerce")
    values = parsed.astype("int64") / 1_000_000_000
    values = values.where(parsed.notna())
    return values.astype("float64")


def _align_bounds_to_series(start: float, end: float, series) -> tuple[float, float]:
    values = series.dropna()
    if values.empty:
        return start, end
    median = float(values.abs().median())
    bound = max(abs(start), abs(end))
    if median > 1e15 and bound < 1e12:
        return start * 1_000_000_000, end * 1_000_000_000
    if median > 1e12 and bound < 1e10:
        return start * 1000, end * 1000
    if median < 1e11 and bound > 1e15:
        return start / 1_000_000_000, end / 1_000_000_000
    if median < 1e11 and bound > 1e12:
        return start / 1000, end / 1000
    return start, end


def _pd():
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - depends on optional research deps
        raise RuntimeError(
            "ShardRCA telemetry cataloging requires pandas. Install requirements-research.txt."
        ) from exc
    return pd

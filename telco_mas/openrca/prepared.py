"""Immutable, label-safe prepared telemetry cache for OpenRCA Telecom."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .dataset import OpenRCADataset
from .task_parser import ParsedOpenRCATask, parse_all_runtime_tasks

PREPARED_FORMAT_VERSION = 1
METRIC_FILES = (
    "metric_node.csv",
    "metric_container.csv",
    "metric_service.csv",
    "metric_middleware.csv",
    "metric_app.csv",
)
TRACE_FILE = "trace_span.csv"


class PreparedOpenRCAError(RuntimeError):
    pass


class PreparedOpenRCA:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "manifest.json"
        if not self.manifest_path.exists():
            raise PreparedOpenRCAError(f"Missing prepared OpenRCA manifest: {self.manifest_path}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format_version") != PREPARED_FORMAT_VERSION:
            raise PreparedOpenRCAError("Unsupported prepared OpenRCA format version")
        if self.manifest.get("status") != "ready":
            raise PreparedOpenRCAError("Prepared OpenRCA cache is not ready")

    def row_dir(self, row_id: int) -> Path:
        path = self.root / "rows" / f"{int(row_id):03d}"
        if not path.exists():
            raise PreparedOpenRCAError(f"Prepared OpenRCA row is missing: {row_id}")
        return path

    def task(self, row_id: int) -> dict[str, Any]:
        return json.loads((self.row_dir(row_id) / "task.json").read_text(encoding="utf-8"))

    def volume(self, row_id: int) -> dict[str, Any]:
        return json.loads((self.row_dir(row_id) / "volume.json").read_text(encoding="utf-8"))

    def metric_paths(self, row_id: int) -> list[Path]:
        return sorted((self.row_dir(row_id) / "metric").glob("*.parquet"))

    def trace_path(self, row_id: int) -> Path | None:
        path = self.row_dir(row_id) / "trace" / "trace_span.parquet"
        return path if path.exists() else None

    def trace_summary_path(self, row_id: int) -> Path | None:
        path = self.row_dir(row_id) / "trace_summary.parquet"
        return path if path.exists() else None

    def trace_edges_path(self, row_id: int) -> Path | None:
        path = self.row_dir(row_id) / "trace_edges.parquet"
        return path if path.exists() else None

    def metric_stats_path(self, row_id: int) -> Path:
        date_key = str(self.task(row_id)["date_key"])
        return self.root / "dates" / date_key / "metric_stats.parquet"

    def runtime_components(self, row_id: int) -> list[str]:
        """Return candidate components observed in the prepared row telemetry.

        This intentionally derives the component universe from runtime-visible
        telemetry slices, not from ``query.csv`` labels or scoring points.
        """
        pd, _, _ = _dependencies()
        components: set[str] = set()
        for path in self.metric_paths(row_id):
            try:
                columns = set(_parquet_columns(path))
                component_cols = [c for c in ("cmdb_id", "component") if c in columns]
                if not component_cols:
                    continue
                frame = pd.read_parquet(path, columns=component_cols)
            except Exception:
                continue
            for column in component_cols:
                values = frame[column].dropna().astype(str)
                components.update(
                    value.strip()
                    for value in values.unique().tolist()
                    if _looks_like_openrca_component(value)
                )
        summary = self.trace_summary_path(row_id)
        if summary is not None:
            try:
                columns = set(_parquet_columns(summary))
                component_cols = [c for c in ("cmdb_id",) if c in columns]
                frame = pd.read_parquet(summary, columns=component_cols)
            except Exception:
                frame = pd.DataFrame()
            for column in frame.columns:
                values = frame[column].dropna().astype(str)
                components.update(
                    value.strip()
                    for value in values.unique().tolist()
                    if _looks_like_openrca_component(value)
                )
        edges = self.trace_edges_path(row_id)
        if edges is not None:
            try:
                frame = pd.read_parquet(edges, columns=["parent_component", "child_component"])
            except Exception:
                frame = pd.DataFrame()
            for column in frame.columns:
                values = frame[column].dropna().astype(str)
                components.update(
                    value.strip()
                    for value in values.unique().tolist()
                    if _looks_like_openrca_component(value)
                )
        return sorted(components)

    def build_catalog(self, row_id: int):
        from ..shardrca.catalog import TelemetryCatalog, _file_info

        row_dir = self.row_dir(row_id)
        task = self.task(row_id)
        paths = self.metric_paths(row_id)
        trace = self.trace_path(row_id)
        if trace is not None:
            paths.append(trace)
        return TelemetryCatalog(
            dataset="OpenRCA-Telecom-prepared",
            case_id=str(row_id),
            root=str(row_dir),
            query_time=(float(task["start_ms"]) + float(task["end_ms"])) / 2.0,
            time_range_start=float(task["start_ms"]),
            time_range_end=float(task["end_ms"]),
            files=[_file_info(path, row_dir, compute_ranges=False) for path in paths],
            metadata={
                "prepared_manifest_sha256": _sha256_file(self.manifest_path),
                "row_id": row_id,
                "volume": self.volume(row_id),
                "candidate_components": [
                    *[f"os_{index:03d}" for index in range(1, 23)],
                    *[f"docker_{index:03d}" for index in range(1, 9)],
                    *[f"db_{index:03d}" for index in range(1, 14)],
                ],
            },
        )

    def validate_against(self, dataset: OpenRCADataset) -> None:
        source = self.manifest.get("source", {})
        if source.get("query_sha256") != _sha256_file(dataset.query_path):
            raise PreparedOpenRCAError("Prepared cache query.csv hash mismatch")
        current = _telemetry_manifest(dataset.telemetry_dir)
        if source.get("telemetry_manifest_sha256") != current["sha256"]:
            raise PreparedOpenRCAError("Prepared cache telemetry manifest mismatch")


def prepare_dataset(
    dataset: OpenRCADataset,
    out_dir: str | Path,
    *,
    chunksize: int = 100_000,
    force: bool = False,
) -> dict[str, Any]:
    """Build the cache using one bounded pass per source file."""

    pd, pa, pq = _dependencies()
    out = Path(out_dir).expanduser().resolve()
    tasks = parse_all_runtime_tasks(dataset)
    source_manifest = _source_manifest(dataset)
    if out.exists() and not force:
        try:
            prepared = PreparedOpenRCA(out)
            prepared.validate_against(dataset)
            return prepared.manifest
        except PreparedOpenRCAError:
            raise PreparedOpenRCAError(
                f"Prepared directory exists but is stale or incomplete: {out}; rerun with --force"
            )

    tmp = out.with_name(f"{out.name}.building")
    if force:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    started = time.time()
    by_date: dict[str, list[ParsedOpenRCATask]] = defaultdict(list)
    for task in tasks:
        by_date[task.date_key].append(task)
        row_dir = tmp / "rows" / f"{task.row_id:03d}"
        (row_dir / "metric").mkdir(parents=True, exist_ok=True)
        (row_dir / "trace").mkdir(parents=True, exist_ok=True)
        _write_json(row_dir / "task.json", task.runtime_dict())

    date_stats: dict[str, dict[str, Any]] = {}
    for date_key, date_tasks in sorted(by_date.items()):
        date_dir = dataset.telemetry_dir / date_key
        if not date_dir.exists():
            raise PreparedOpenRCAError(f"Missing telemetry date directory: {date_dir}")
        current = {"source_bytes": 0, "metric_rows": 0, "trace_rows": 0}
        metric_stats = []
        for filename in METRIC_FILES:
            source = date_dir / "metric" / filename
            if not source.exists():
                continue
            current["source_bytes"] += source.stat().st_size
            frame = pd.read_csv(source)
            current["metric_rows"] += int(len(frame))
            time_col = _time_column(frame.columns)
            if time_col is None:
                continue
            times = pd.to_numeric(frame[time_col], errors="coerce")
            component_col, metric_col, value_col = _metric_columns(frame.columns, filename)
            if component_col and metric_col and value_col:
                local = frame[[component_col, metric_col, value_col]].copy()
                local[value_col] = pd.to_numeric(local[value_col], errors="coerce")
                local = local.dropna(subset=[component_col, metric_col, value_col])
                if not local.empty:
                    grouped = local.groupby([component_col, metric_col], dropna=True)[value_col]
                    stats = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
                    quantiles = grouped.quantile([0.05, 0.15, 0.90, 0.95]).unstack().reset_index()
                    quantiles.columns = [component_col, metric_col, "p05", "p15", "p90", "p95"]
                    stats = stats.merge(quantiles, on=[component_col, metric_col], how="left")
                    stats = stats.rename(columns={component_col: "component", metric_col: "metric"})
                    stats["source_file"] = filename
                    metric_stats.append(stats)
            elif filename == "metric_app.csv" and "serviceName" in frame:
                for app_metric in ("avg_time", "num", "succee_num", "succee_rate"):
                    if app_metric not in frame:
                        continue
                    local = frame[["serviceName", app_metric]].copy()
                    local[app_metric] = pd.to_numeric(local[app_metric], errors="coerce")
                    local = local.dropna()
                    if local.empty:
                        continue
                    grouped = local.groupby("serviceName", dropna=True)[app_metric]
                    stats = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
                    quantiles = grouped.quantile([0.05, 0.15, 0.90, 0.95]).unstack().reset_index()
                    quantiles.columns = ["serviceName", "p05", "p15", "p90", "p95"]
                    stats = stats.merge(quantiles, on="serviceName", how="left")
                    stats = stats.rename(columns={"serviceName": "component"})
                    stats["metric"] = app_metric
                    stats["source_file"] = filename
                    metric_stats.append(stats)
            for task in date_tasks:
                mask = times.between(task.start_ms, task.end_ms, inclusive="both")
                sliced = frame.loc[mask]
                if not sliced.empty:
                    destination = tmp / "rows" / f"{task.row_id:03d}" / "metric" / filename.replace(".csv", ".parquet")
                    sliced.to_parquet(destination, index=False, compression="zstd")

        stats_frame = pd.concat(metric_stats, ignore_index=True) if metric_stats else pd.DataFrame()
        stats_path = tmp / "dates" / date_key / "metric_stats.parquet"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_frame.to_parquet(stats_path, index=False, compression="zstd")

        trace_source = date_dir / "trace" / TRACE_FILE
        if trace_source.exists():
            current["source_bytes"] += trace_source.stat().st_size
            writers: dict[int, Any] = {}
            try:
                for chunk in pd.read_csv(trace_source, chunksize=chunksize):
                    current["trace_rows"] += int(len(chunk))
                    chunk = _normalize_trace_frame(chunk, pd=pd)
                    time_col = _time_column(chunk.columns)
                    if time_col is None:
                        continue
                    times = pd.to_numeric(chunk[time_col], errors="coerce")
                    for task in date_tasks:
                        sliced = chunk.loc[times.between(task.start_ms, task.end_ms, inclusive="both")]
                        if sliced.empty:
                            continue
                        table = pa.Table.from_pandas(sliced, preserve_index=False)
                        writer = writers.get(task.row_id)
                        if writer is None:
                            destination = tmp / "rows" / f"{task.row_id:03d}" / "trace" / "trace_span.parquet"
                            writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                            writers[task.row_id] = writer
                        writer.write_table(table)
            finally:
                for writer in writers.values():
                    writer.close()
        date_stats[date_key] = current

    row_volumes = {}
    for task in tasks:
        row_dir = tmp / "rows" / f"{task.row_id:03d}"
        volume = _finalize_row(row_dir, pd=pd)
        row_volumes[str(task.row_id)] = volume
        _write_json(row_dir / "volume.json", volume)

    volume_bins = _volume_bins(row_volumes)
    manifest = {
        "format_version": PREPARED_FORMAT_VERSION,
        "status": "ready",
        "created_at": int(time.time()),
        "source": source_manifest,
        "row_count": len(tasks),
        "row_ids": [task.row_id for task in tasks],
        "dates": sorted(by_date),
        "date_stats": date_stats,
        "row_volumes": row_volumes,
        "volume_bins": volume_bins,
        "build": {
            "chunksize": chunksize,
            "elapsed_s": round(time.time() - started, 3),
            "prepared_bytes": _directory_bytes(tmp),
        },
    }
    _write_json(tmp / "manifest.json", manifest)
    os.replace(tmp, out)
    return manifest


def _finalize_row(row_dir: Path, *, pd) -> dict[str, Any]:
    rows_by_modality = {"metrics": 0, "traces": 0}
    bytes_by_modality = {"metrics": 0, "traces": 0}
    for path in sorted((row_dir / "metric").glob("*.parquet")):
        rows_by_modality["metrics"] += _parquet_rows(path)
        bytes_by_modality["metrics"] += path.stat().st_size
    trace_path = row_dir / "trace" / "trace_span.parquet"
    if trace_path.exists():
        rows_by_modality["traces"] = _parquet_rows(trace_path)
        bytes_by_modality["traces"] = trace_path.stat().st_size
        frame = pd.read_parquet(
            trace_path,
            columns=[
                column
                for column in ("id", "pid", "cmdb_id", "serviceName", "dsName", "callType", "startTime", "elapsedTime", "success")
                if column in _parquet_columns(trace_path)
            ],
        )
        if not frame.empty:
            group_cols = [column for column in ("cmdb_id", "serviceName", "dsName", "callType") if column in frame]
            if group_cols and "elapsedTime" in frame:
                frame["elapsedTime"] = pd.to_numeric(frame["elapsedTime"], errors="coerce")
                summary = frame.groupby(group_cols, dropna=False)["elapsedTime"].agg(
                    ["count", "mean", "median", "max"]
                ).reset_index()
                if "startTime" in frame:
                    frame["startTime"] = pd.to_numeric(frame["startTime"], errors="coerce")
                    first_start = (
                        frame.groupby(group_cols, dropna=False)["startTime"]
                        .min()
                        .rename("first_start")
                        .reset_index()
                    )
                    valid_elapsed = frame.dropna(subset=["elapsedTime"])
                    if not valid_elapsed.empty:
                        max_indices = valid_elapsed.groupby(group_cols, dropna=False)["elapsedTime"].idxmax()
                        max_start = (
                            valid_elapsed.loc[max_indices, [*group_cols, "startTime"]]
                            .rename(columns={"startTime": "max_elapsed_start"})
                        )
                        summary = summary.merge(max_start, on=group_cols, how="left")
                    summary = summary.merge(first_start, on=group_cols, how="left")
                if "success" in frame:
                    failed = frame["success"].astype(str).str.lower().isin({"false", "0", "fail", "failed"})
                    failure_counts = (
                        frame.assign(_failed=failed.astype(int))
                        .groupby(group_cols, dropna=False)["_failed"]
                        .sum()
                        .rename("failure_count")
                        .reset_index()
                    )
                    summary = summary.merge(failure_counts, on=group_cols, how="left")
                summary.to_parquet(row_dir / "trace_summary.parquet", index=False, compression="zstd")
            if {"id", "pid", "cmdb_id"}.issubset(frame.columns):
                nodes = frame[["id", "cmdb_id"]].dropna().drop_duplicates("id")
                nodes = nodes.rename(columns={"id": "pid", "cmdb_id": "parent_component"})
                edges = frame[["pid", "cmdb_id"]].dropna().rename(columns={"cmdb_id": "child_component"})
                edges = edges.merge(nodes, on="pid", how="inner")
                edges = edges[edges["parent_component"] != edges["child_component"]]
                if not edges.empty:
                    edge_summary = (
                        edges.groupby(["parent_component", "child_component"], dropna=False)
                        .size()
                        .rename("span_count")
                        .reset_index()
                        .sort_values("span_count", ascending=False)
                    )
                    edge_summary.to_parquet(
                        row_dir / "trace_edges.parquet",
                        index=False,
                        compression="zstd",
                    )
    total_rows = rows_by_modality["metrics"] + rows_by_modality["traces"]
    total_bytes = bytes_by_modality["metrics"] + bytes_by_modality["traces"]
    return {
        "rows": rows_by_modality,
        "bytes": bytes_by_modality,
        "total_rows": total_rows,
        "total_bytes": total_bytes,
    }


def _source_manifest(dataset: OpenRCADataset) -> dict[str, Any]:
    telemetry = _telemetry_manifest(dataset.telemetry_dir)
    return {
        "query_path": str(dataset.query_path),
        "query_sha256": _sha256_file(dataset.query_path),
        "telemetry_dir": str(dataset.telemetry_dir),
        "telemetry_manifest_sha256": telemetry["sha256"],
        "telemetry_file_count": telemetry["file_count"],
        "telemetry_total_bytes": telemetry["total_bytes"],
    }


def _volume_bins(volumes: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    ordered = sorted((int(row_id), int(item["total_rows"])) for row_id, item in volumes.items())
    ordered.sort(key=lambda item: (item[1], item[0]))
    n = len(ordered)
    low_end = n // 3
    high_start = n - n // 3
    return {
        "low": sorted(row_id for row_id, _ in ordered[:low_end]),
        "middle": sorted(row_id for row_id, _ in ordered[low_end:high_start]),
        "high": sorted(row_id for row_id, _ in ordered[high_start:]),
    }


def _metric_columns(columns: Iterable[str], filename: str) -> tuple[str | None, str | None, str | None]:
    names = set(str(column) for column in columns)
    if filename == "metric_app.csv":
        return (
            "serviceName" if "serviceName" in names else None,
            None,
            None,
        )
    component = next((name for name in ("cmdb_id", "component", "serviceName") if name in names), None)
    metric = next((name for name in ("name", "metric", "item") if name in names), None)
    value = next((name for name in ("value", "avg_time", "succee_rate") if name in names), None)
    return component, metric, value


def _normalize_trace_frame(frame, *, pd):
    numeric = {"startTime", "elapsedTime"}
    for column in frame.columns:
        if column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = frame[column].astype("string")
    return frame


def _time_column(columns: Iterable[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in columns}
    for name in ("timestamp", "starttime", "starttimemillis", "time", "datetime"):
        if name in lower:
            return lower[name]
    return None


def _parquet_rows(path: Path) -> int:
    _, _, pq = _dependencies()
    return int(pq.ParquetFile(path).metadata.num_rows)


def _parquet_columns(path: Path) -> list[str]:
    _, _, pq = _dependencies()
    return list(pq.ParquetFile(path).schema.names)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _telemetry_manifest(root: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(path.relative_to(root)), "bytes": size})
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_openrca_component(value: Any) -> bool:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "<na>", "none"}:
        return False
    return text.startswith(("os_", "docker_", "db_"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _dependencies():
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - optional research dependency
        raise PreparedOpenRCAError(
            "Prepared OpenRCA cache requires pandas and pyarrow; install requirements-research.txt"
        ) from exc
    return pd, pa, pq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare label-safe OpenRCA telemetry windows.")
    parser.add_argument("--data-dir", default=os.getenv("OPENRCA_DATA_DIR") or "data/openrca")
    parser.add_argument("--dataset", default="Telecom")
    parser.add_argument("--out", default="data/openrca_prepared/Telecom")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    dataset = OpenRCADataset(args.data_dir, dataset=args.dataset)
    manifest = prepare_dataset(dataset, args.out, chunksize=args.chunksize, force=args.force)
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "rows": manifest["row_count"],
        "prepared_bytes": manifest["build"]["prepared_bytes"],
        "elapsed_s": manifest["build"]["elapsed_s"],
        "volume_bins": {name: len(ids) for name, ids in manifest["volume_bins"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

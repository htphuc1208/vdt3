"""Bounded telemetry tools for OpenRCA Telecom data."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from .dataset import OpenRCADataset
from .prepared import PreparedOpenRCA

TELECOM_REASONS = ["CPU fault", "network delay", "network loss", "db connection limit", "db close"]
TELECOM_COMPONENTS = [
    *[f"os_{index:03d}" for index in range(1, 23)],
    *[f"docker_{index:03d}" for index in range(1, 9)],
    *[f"db_{index:03d}" for index in range(1, 14)],
]
UTC_PLUS_8 = ZoneInfo("Asia/Shanghai")


def candidate_catalog_for_row(prepared: PreparedOpenRCA, row_id: int) -> dict[str, Any]:
    """Build a label-safe OpenRCA candidate catalog from prepared runtime telemetry."""

    components = prepared.runtime_components(row_id)
    source = "prepared_runtime_telemetry"
    fallback_used = False
    if not components:
        # Empty fixture rows and some partial smoke caches have no observable
        # runtime components. This is a protocol prior fallback, never a label
        # lookup; confirmatory rows should normally use prepared telemetry.
        components = list(TELECOM_COMPONENTS)
        source = "protocol_prior_no_runtime_components"
        fallback_used = True
    return {
        "components": sorted(dict.fromkeys(components)),
        "reasons": list(TELECOM_REASONS),
        "source": {
            "components": source,
            "reasons": "protocol_prior_openrca_reason_catalog",
            "label_derived": False,
            "fallback_used": fallback_used,
        },
    }


class OpenRCATelemetryTools:
    def __init__(self, dataset: OpenRCADataset, *, max_rows_per_file: int = 200_000) -> None:
        self.dataset = dataset
        self.max_rows_per_file = max_rows_per_file

    def get_candidate_catalog(self) -> dict[str, Any]:
        return {
            "components": TELECOM_COMPONENTS,
            "reasons": TELECOM_REASONS,
            "telemetry_dates": self.dataset.available_dates(),
            "timezone": "UTC+8 / Asia/Shanghai",
        }

    def summarize_metric_anomalies(
        self,
        start_datetime: str | None,
        end_datetime: str | None,
        metric_file: str | None = None,
        components: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        window = self._window(start_datetime, end_datetime)
        if "error" in window:
            return {"error": window["error"], "anomalies": []}
        files = self._telemetry_files("metric", window["start_ms"], window["end_ms"], metric_file)
        component_filter = set(components or [])
        rows = []
        for path in files:
            for row in self._metric_rows(path):
                if component_filter and row["component"] not in component_filter:
                    continue
                if window["start_ms"] <= row["timestamp_ms"] <= window["end_ms"]:
                    rows.append(row)
        if not rows:
            return {
                "time_range": window["label"],
                "files_scanned": len(files),
                "anomalies": [],
                "telemetry_gaps": ["No metric samples found in the requested window."],
            }
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            grouped.setdefault((row["component"], row["metric"]), []).append(row)
        anomalies = []
        for (component, metric), group in grouped.items():
            values = [g["value"] for g in group]
            avg = sum(values) / len(values)
            baseline = self._baseline(component, metric, files)
            if baseline is None:
                continue
            deviation = abs(avg - baseline) / (abs(baseline) + 1.0)
            if deviation > 0:
                anomalies.append({
                    "component": component,
                    "metric": metric,
                    "avg_value": round(avg, 6),
                    "baseline": round(baseline, 6),
                    "direction": "high" if avg > baseline else "low",
                    "severity": round(deviation, 6),
                })
        anomalies.sort(key=lambda item: item["severity"], reverse=True)
        return {
            "time_range": window["label"],
            "files_scanned": len(files),
            "anomalies": anomalies[: max(1, min(limit, 100))],
            "telemetry_gaps": [] if anomalies else ["No metric anomalies found in the requested window."],
        }

    def search_logs(
        self,
        start_datetime: str | None,
        end_datetime: str | None,
        keyword: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        window = self._window(start_datetime, end_datetime)
        if "error" in window:
            return {"error": window["error"], "matches": []}
        needle = (keyword or "").lower().strip()
        matches = []
        files_scanned = 0
        for date_dir in self._date_dirs_for_window(window["start_ms"], window["end_ms"]):
            log_dir = date_dir / "log"
            if not log_dir.exists():
                continue
            for path in sorted(p for p in log_dir.rglob("*") if p.is_file()):
                files_scanned += 1
                for line in path.read_text(errors="ignore").splitlines()[: self.max_rows_per_file]:
                    if not needle or needle in line.lower():
                        matches.append({"file": path.name, "line": line[:500]})
                    if len(matches) >= limit:
                        break
                if len(matches) >= limit:
                    break
        return {"time_range": window["label"], "files_scanned": files_scanned, "keyword": keyword, "matches": matches}

    def _metric_rows(self, path: Path) -> Iterable[dict[str, Any]]:
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if idx >= self.max_rows_per_file:
                    break
                metric = row.get("name") or row.get("metric") or row.get("item") or ""
                component = row.get("cmdb_id") or row.get("component") or row.get("service") or ""
                timestamp = row.get("timestamp") or row.get("time") or ""
                value = _to_float(row.get("value"))
                timestamp_ms = _to_ms(timestamp)
                if metric and component and value is not None and timestamp_ms is not None:
                    yield {"component": component, "metric": metric, "timestamp_ms": timestamp_ms, "value": value}

    def _baseline(self, component: str, metric: str, files: list[Path]) -> float | None:
        values = []
        for path in files[:2]:
            for row in self._metric_rows(path):
                if row["component"] == component and row["metric"] == metric:
                    values.append(row["value"])
                if len(values) >= 200:
                    break
        if not values:
            return None
        return sum(values) / len(values)

    def _telemetry_files(self, kind: str, start_ms: int, end_ms: int, specific: str | None = None) -> list[Path]:
        files = []
        for date_dir in self._date_dirs_for_window(start_ms, end_ms):
            folder = date_dir / kind
            if not folder.exists():
                continue
            if specific:
                path = folder / specific
                if path.exists():
                    files.append(path)
            else:
                files.extend(sorted(path for path in folder.glob("*.csv") if path.is_file()))
        return files

    def _date_dirs_for_window(self, start_ms: int, end_ms: int) -> list[Path]:
        out = []
        for date_dir in self.dataset.telemetry_date_dirs():
            try:
                date = datetime.strptime(date_dir.name, "%Y_%m_%d").replace(tzinfo=UTC_PLUS_8)
            except ValueError:
                out.append(date_dir)
                continue
            day_start = int(date.timestamp() * 1000)
            day_end = day_start + 24 * 60 * 60 * 1000
            if day_start <= end_ms and day_end >= start_ms:
                out.append(date_dir)
        return out

    def _window(self, start_datetime: str | None, end_datetime: str | None) -> dict[str, Any]:
        start = _parse_dt(start_datetime)
        end = _parse_dt(end_datetime)
        if start is None or end is None:
            return {"error": "start_datetime and end_datetime must use '%Y-%m-%d %H:%M:%S' in UTC+8"}
        return {
            "start_ms": int(start.timestamp() * 1000),
            "end_ms": int(end.timestamp() * 1000),
            "label": f"{start_datetime} -> {end_datetime}",
        }


def _parse_dt(text: str | None):
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC_PLUS_8)
    except ValueError:
        return None


def _to_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        num = float(value)
        return int(num if num > 10_000_000_000 else num * 1000)
    except ValueError:
        parsed = _parse_dt(value)
        return int(parsed.timestamp() * 1000) if parsed else None


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None

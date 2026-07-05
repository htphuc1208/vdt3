"""Deterministic, label-free KPI summaries shared by every TelecomTS system."""
from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, pstdev
from typing import Any


KPI_SHARDS = {
    "radio_quality": (
        "RSRP",
        "UL_SNR",
        "DL_BLER",
        "UL_BLER",
        "DL_MCS",
        "UL_MCS",
    ),
    "resource_capacity": (
        "UL_NPRB",
        "Estimated_UL_Buffer",
        "PRBs_DL_Current",
        "PRBs_UL_Current",
        "PRB_Utilization_DL",
        "PRB_Utilization_UL",
    ),
    "traffic_protocol": (
        "TX_Bytes",
        "RX_Bytes",
        "UL_NumberOfPackets",
        "DL_NumberOfPackets",
        "UL_Protocol",
        "DL_Protocol",
    ),
}
PROTOTYPE_METRIC_KEYS = (
    "standardized_edge_delta",
    "standardized_full_slope",
    "largest_step_std",
    "direction_change_rate",
    "zero_fraction",
    "extreme_repeat_fraction",
)

BASE_CLASS_CATALOG = {
    "Antenna Failure": {
        "domain": "hardware",
        "temporal": True,
        "mechanism": "antenna or RF front-end path degradation before scheduler/load effects",
        "primary_evidence": ["RSRP/SNR degradation", "BLER increase", "MCS drop"],
        "distinguishers": [
            "radio-quality failure can create downstream PRB, buffer, and traffic symptoms",
            "do not require high offered load or protocol burst as the initiating cause",
        ],
    },
    "Co-Channel Interference (Mild)": {
        "domain": "infrastructure",
        "temporal": False,
        "mechanism": "persistent external or neighboring-cell interference with limited severity",
        "primary_evidence": ["moderate BLER/SNR degradation", "capacity pressure without collapse"],
        "distinguishers": [
            "less extreme than severe co-channel interference",
            "not primarily a scheduler/allocation control fault",
        ],
    },
    "Co-Channel Interference (Severe)": {
        "domain": "infrastructure",
        "temporal": False,
        "mechanism": "strong persistent same-channel interference degrading the radio link",
        "primary_evidence": ["large BLER/SNR degradation", "MCS and throughput collapse"],
        "distinguishers": [
            "more severe and cross-layer than mild interference",
            "radio impairment is primary rather than traffic demand",
        ],
    },
    "Faulty RF Filters (Temporal)": {
        "domain": "hardware",
        "temporal": True,
        "mechanism": "time-local RF filtering impairment distorting radio quality",
        "primary_evidence": ["temporal BLER/SNR/MCS disturbance", "radio symptoms concentrated in time"],
        "distinguishers": [
            "more transient than antenna failure",
            "does not require allocation mismatch as the initiating cause",
        ],
    },
    "Doppler Shift (Severe)": {
        "domain": "environment",
        "temporal": True,
        "mechanism": "mobility-induced frequency shift causing fast channel variation",
        "primary_evidence": ["rapid radio-quality variation", "MCS/SNR instability"],
        "distinguishers": [
            "look for oscillatory or mobility-like radio effects",
            "not explained by static load or buffer growth alone",
        ],
    },
    "Faulty Handover Algorithm (Too Frequent)": {
        "domain": "software",
        "temporal": True,
        "mechanism": "control-plane handover churn causing repeated service disturbance",
        "primary_evidence": ["bursty cross-layer interruptions", "radio and traffic instability"],
        "distinguishers": [
            "symptoms recur in episodes rather than one pure load ramp",
            "not a pure RF hardware path failure",
        ],
    },
    "Buffer Overflow (Gradual Buildup)": {
        "domain": "software_or_infrastructure",
        "temporal": True,
        "mechanism": "queued uplink/downlink demand accumulates before throughput degrades",
        "primary_evidence": ["buffer growth", "packet/byte backlog", "eventual throughput loss"],
        "distinguishers": [
            "queue buildup is primary, not merely a consequence of weak radio",
            "more queue-centric than broad network congestion",
        ],
    },
    "Resource Allocation Bugs": {
        "domain": "software",
        "temporal": True,
        "mechanism": "scheduler allocates PRBs inconsistently with demand or channel state",
        "primary_evidence": ["PRB allocation/utilization mismatch", "stable radio with capacity anomalies"],
        "distinguishers": [
            "look for allocation-control inconsistency rather than high user demand",
            "traffic/resource symptoms can occur without antenna-like radio failure",
        ],
    },
    "High Network Congestion (Gradual Buildup)": {
        "domain": "usage",
        "temporal": True,
        "mechanism": "offered load increases over time until shared resources saturate",
        "primary_evidence": ["gradual PRB utilization increase", "buffer/packet/byte load buildup"],
        "distinguishers": [
            "load ramp is primary; radio impairment is secondary if present",
            "not a sudden isolated spike",
        ],
    },
    "High Network Congestion (Sudden Spike)": {
        "domain": "usage",
        "temporal": False,
        "mechanism": "abrupt offered-load spike saturating shared resources",
        "primary_evidence": ["sharp PRB or traffic burst", "short-lived load surge"],
        "distinguishers": [
            "sudden load evidence should dominate over gradual ramps",
            "radio KPIs may stay stable compared with hardware/interference faults",
        ],
    },
}


def summarize_event(
    payload: dict[str, Any],
    *,
    class_catalog: dict[str, Any] | None = None,
    include_prototype_matches: bool = False,
) -> dict[str, Any]:
    """Return one complete board of summaries without using evaluator-only fields."""
    kpis = payload.get("kpis")
    if not isinstance(kpis, dict):
        raise ValueError("TelecomTS payload has no KPI mapping")
    catalog = class_catalog or BASE_CLASS_CATALOG
    shards = {
        shard: {
            name: summarize_series(kpis[name])
            for name in names
        }
        for shard, names in KPI_SHARDS.items()
    }
    return {
        "sample_length": int(payload.get("sample_length") or 0),
        "sampling_rate_hz": int(payload.get("sampling_rate_hz") or 0),
        "scenario": dict(payload.get("scenario") or {}),
        "class_catalog": _runtime_class_catalog(catalog),
        "prototype_matches": (
            _prototype_matches(shards, catalog)
            if include_prototype_matches
            else _empty_prototype_matches()
        ),
        "shards": shards,
    }


def summarize_series(values: list[Any], *, bins: int = 16) -> dict[str, Any]:
    if not isinstance(values, list) or not values:
        raise ValueError("KPI series must be a non-empty list")
    if all(value is None or isinstance(value, str) for value in values):
        return _categorical_summary(values)
    numeric = [float(value) for value in values]
    n = len(numeric)
    mean = fmean(numeric)
    std = pstdev(numeric) if n > 1 else 0.0
    edge = min(16, max(1, n // 4))
    first_mean = fmean(numeric[:edge])
    last_mean = fmean(numeric[-edge:])
    scale = std if std > 1e-12 else 1.0
    diffs = [numeric[index] - numeric[index - 1] for index in range(1, n)]
    nonzero_signs = [1 if value > 0 else -1 for value in diffs if abs(value) > 1e-12]
    sign_changes = sum(
        1 for left, right in zip(nonzero_signs, nonzero_signs[1:]) if left != right
    )
    slope = _linear_slope(numeric)
    ordered = sorted(numeric)
    minimum = ordered[0]
    maximum = ordered[-1]
    extreme_count = sum(1 for value in numeric if value in {minimum, maximum})
    bin_means = _bin_means(numeric, bins)
    peak_bin_index = max(range(len(bin_means)), key=lambda index: bin_means[index])
    trough_bin_index = min(range(len(bin_means)), key=lambda index: bin_means[index])
    edge_high = max(first_mean, last_mean)
    edge_low = min(first_mean, last_mean)
    mid_peak_std = (bin_means[peak_bin_index] - edge_high) / scale
    mid_dip_std = (edge_low - bin_means[trough_bin_index]) / scale
    largest_step_std = max((abs(value) for value in diffs), default=0.0) / scale
    direction_change_rate = sign_changes / max(1, len(nonzero_signs) - 1)
    zero_fraction = sum(1 for value in numeric if value == 0.0) / n
    extreme_repeat_fraction = extreme_count / n
    edge_delta = (last_mean - first_mean) / scale
    full_slope = slope * max(1, n - 1) / scale
    return {
        "kind": "numeric",
        "n": n,
        "mean": _round(mean),
        "std": _round(std),
        "min": _round(minimum),
        "q10": _round(_quantile(ordered, 0.10)),
        "median": _round(_quantile(ordered, 0.50)),
        "q90": _round(_quantile(ordered, 0.90)),
        "max": _round(maximum),
        "first_mean": _round(first_mean),
        "last_mean": _round(last_mean),
        "standardized_edge_delta": _round(edge_delta),
        "standardized_full_slope": _round(full_slope),
        "largest_step_std": _round(largest_step_std),
        "direction_change_rate": _round(direction_change_rate),
        "zero_fraction": _round(zero_fraction),
        "extreme_repeat_fraction": _round(extreme_repeat_fraction),
        "peak_bin_index": peak_bin_index,
        "trough_bin_index": trough_bin_index,
        "mid_peak_std": _round(mid_peak_std),
        "mid_dip_std": _round(mid_dip_std),
        "shape_tags": _shape_tags(
            edge_delta=edge_delta,
            full_slope=full_slope,
            largest_step_std=largest_step_std,
            direction_change_rate=direction_change_rate,
            zero_fraction=zero_fraction,
            extreme_repeat_fraction=extreme_repeat_fraction,
            mid_peak_std=mid_peak_std,
            mid_dip_std=mid_dip_std,
            peak_bin_index=peak_bin_index,
            trough_bin_index=trough_bin_index,
            bin_count=len(bin_means),
        ),
        "bin_means": [_round(value) for value in bin_means],
    }


def _categorical_summary(values: list[Any]) -> dict[str, Any]:
    normalized = ["None" if value is None else str(value) for value in values]
    counts = Counter(normalized)
    transitions = sum(1 for left, right in zip(normalized, normalized[1:]) if left != right)
    return {
        "kind": "categorical",
        "n": len(normalized),
        "fractions": {
            key: _round(count / len(normalized))
            for key, count in sorted(counts.items())
        },
        "transition_rate": _round(transitions / max(1, len(normalized) - 1)),
    }


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = fmean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    return numerator / denominator if denominator else 0.0


def _quantile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bin_means(values: list[float], bins: int) -> list[float]:
    count = min(max(1, bins), len(values))
    out = []
    for index in range(count):
        start = round(index * len(values) / count)
        end = round((index + 1) * len(values) / count)
        out.append(fmean(values[start:end]))
    return out


def _round(value: float) -> float:
    return float(f"{value:.6g}")


def _runtime_class_catalog(class_catalog: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for class_name, metadata in class_catalog.items():
        if not isinstance(metadata, dict):
            sanitized[class_name] = metadata
            continue
        sanitized[class_name] = {
            key: value
            for key, value in metadata.items()
            if key != "shape_prototype"
        }
    return sanitized


def _prototype_matches(
    shards: dict[str, dict[str, dict[str, Any]]],
    class_catalog: dict[str, Any],
) -> dict[str, Any]:
    class_names = [
        class_name
        for class_name, metadata in class_catalog.items()
        if isinstance(metadata, dict) and isinstance(metadata.get("shape_prototype"), dict)
    ]
    if not class_names:
        return {"overall": [], "by_shard": {name: [] for name in KPI_SHARDS}}
    overall = [
        _prototype_distance_row(class_name, class_catalog[class_name], shards)
        for class_name in class_names
    ]
    by_shard = {
        shard_name: [
            _prototype_distance_row(class_name, class_catalog[class_name], shards, shard=shard_name)
            for class_name in class_names
        ]
        for shard_name in KPI_SHARDS
    }
    return {
        "overall": _rank_matches(overall, limit=len(class_names)),
        "by_shard": {
            shard_name: _rank_matches(rows, limit=5)
            for shard_name, rows in by_shard.items()
        },
    }


def _empty_prototype_matches() -> dict[str, Any]:
    return {"overall": [], "by_shard": {name: [] for name in KPI_SHARDS}}


def _prototype_distance_row(
    class_name: str,
    metadata: dict[str, Any],
    shards: dict[str, dict[str, dict[str, Any]]],
    *,
    shard: str | None = None,
) -> dict[str, Any]:
    prototype = metadata.get("shape_prototype") or {}
    shard_distances = {}
    selected_shards = [shard] if shard else list(KPI_SHARDS)
    for shard_name in selected_shards:
        distance, count = _prototype_distance_for_shard(
            shards.get(shard_name) or {},
            prototype.get(shard_name) or {},
        )
        if count:
            shard_distances[shard_name] = distance
    if not shard_distances:
        distance = 999.0
    else:
        distance = fmean(shard_distances.values())
    return {
        "class": class_name,
        "distance": _round(distance),
        "similarity": _round(1.0 / (1.0 + distance)),
        "support_events": int(metadata.get("prototype_support_events") or 0),
        "shard_distances": {
            name: _round(value)
            for name, value in sorted(shard_distances.items())
        },
    }


def _prototype_distance_for_shard(
    shard: dict[str, dict[str, Any]],
    prototype: dict[str, Any],
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for kpi_name, metric_stats in prototype.items():
        summary = shard.get(kpi_name)
        if not isinstance(summary, dict) or summary.get("kind") != "numeric":
            continue
        if not isinstance(metric_stats, dict):
            continue
        for metric in PROTOTYPE_METRIC_KEYS:
            stats = metric_stats.get(metric)
            value = summary.get(metric)
            if not isinstance(stats, dict) or not isinstance(value, int | float):
                continue
            mean = stats.get("mean")
            std = stats.get("std")
            if not isinstance(mean, int | float) or not isinstance(std, int | float):
                continue
            scale = max(abs(float(std)), 1e-6)
            z_score = abs(float(value) - float(mean)) / scale
            total += min(z_score, 8.0) ** 2
            count += 1
    if not count:
        return 999.0, 0
    return math.sqrt(total / count), count


def _rank_matches(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["distance"], -row["support_events"], row["class"]))[:limit]


def _shape_tags(
    *,
    edge_delta: float,
    full_slope: float,
    largest_step_std: float,
    direction_change_rate: float,
    zero_fraction: float,
    extreme_repeat_fraction: float,
    mid_peak_std: float,
    mid_dip_std: float,
    peak_bin_index: int,
    trough_bin_index: int,
    bin_count: int,
) -> list[str]:
    tags = []
    if edge_delta >= 1.0:
        tags.append("rising_edge")
    elif edge_delta <= -1.0:
        tags.append("falling_edge")
    if full_slope >= 1.0:
        tags.append("upward_trend")
    elif full_slope <= -1.0:
        tags.append("downward_trend")
    if largest_step_std >= 4.0:
        tags.append("abrupt_step")
    if direction_change_rate >= 0.6:
        tags.append("oscillatory")
    if zero_fraction >= 0.2:
        tags.append("zero_plateau")
    if extreme_repeat_fraction >= 0.9:
        tags.append("constant_or_clipped")
    if _is_middle_bin(peak_bin_index, bin_count) and mid_peak_std >= 1.5:
        tags.append("mid_event_spike")
    if _is_middle_bin(trough_bin_index, bin_count) and mid_dip_std >= 1.5:
        tags.append("mid_event_dip")
    if not tags:
        tags.append("stable_or_low_signal")
    return tags


def _is_middle_bin(index: int, bin_count: int) -> bool:
    return bin_count // 4 <= index < (3 * bin_count + 3) // 4

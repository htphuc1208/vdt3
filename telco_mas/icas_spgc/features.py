"""Agent-specific, deterministic summaries of variable-length KPI traces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset import runtime_feature_columns


LEVEL_STATS = ("mean", "min", "max", "median", "q25", "q75", "first", "last")
DYNAMIC_STATS = ("std", "iqr", "range", "mad_diff", "diff_std", "slope")
ROOT_CAUSAL_PREFIXES = (
    ("feature13", "feature15"),
    ("feature13", "feature19", "featureX", "featureY"),
    ("feature20", "feature60", "featureX", "featureY"),
)


@dataclass(frozen=True)
class FeatureViews:
    sample_ids: np.ndarray
    full: pd.DataFrame
    level: pd.DataFrame
    dynamics: pd.DataFrame
    causal: pd.DataFrame
    causal_by_root: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {name: 0.0 for name in LEVEL_STATS + DYNAMIC_STATS}
    differences = np.diff(finite)
    if len(finite) > 1:
        x = np.arange(len(finite), dtype=float)
        slope = float(np.polyfit(x, finite, 1)[0])
    else:
        slope = 0.0
    q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "first": float(finite[0]),
        "last": float(finite[-1]),
        "std": float(np.std(finite)),
        "iqr": float(q75 - q25),
        "range": float(np.max(finite) - np.min(finite)),
        "mad_diff": float(np.mean(np.abs(differences))) if len(differences) else 0.0,
        "diff_std": float(np.std(differences)) if len(differences) else 0.0,
        "slope": slope,
    }


def build_feature_views(observations: pd.DataFrame) -> FeatureViews:
    columns = runtime_feature_columns(observations)
    if "sample_index" not in observations:
        raise ValueError("sample_index is required only for grouping")
    rows: list[dict[str, float]] = []
    ids: list[int] = []
    for sample_id, group in observations.groupby("sample_index", sort=True):
        row: dict[str, float] = {}
        for column in columns:
            summary = _summarize(group[column].to_numpy(dtype=float))
            row.update({f"{column}__{stat}": value for stat, value in summary.items()})
        rows.append(row)
        ids.append(int(sample_id))
    full = pd.DataFrame(rows, index=ids).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    level_columns = [column for column in full if column.rsplit("__", 1)[1] in LEVEL_STATS]
    dynamic_columns = [column for column in full if column.rsplit("__", 1)[1] in DYNAMIC_STATS]
    causal_root_columns = [
        [
            column
            for column in full
            if column.split("__", 1)[0].startswith(prefixes)
        ]
        for prefixes in ROOT_CAUSAL_PREFIXES
    ]
    if any(not columns for columns in causal_root_columns):
        raise ValueError("No preregistered causal-path KPI columns are present")
    causal_columns = list(dict.fromkeys(column for columns in causal_root_columns for column in columns))
    return FeatureViews(
        sample_ids=np.asarray(ids, dtype=int),
        full=full,
        level=full.loc[:, level_columns],
        dynamics=full.loc[:, dynamic_columns],
        causal=full.loc[:, causal_columns],
        causal_by_root=tuple(full.loc[:, columns] for columns in causal_root_columns),
    )

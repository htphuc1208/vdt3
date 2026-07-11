"""Dataset loading with explicit label/identifier separation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import PROTOCOL


@dataclass(frozen=True)
class SplitData:
    observations: pd.DataFrame
    labels: pd.DataFrame
    sample_ids: np.ndarray


def _read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path).set_index("sample_index")
    columns = PROTOCOL["label_columns"]
    missing = set(columns) - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns in {path}: {sorted(missing)}")
    labels = labels.loc[:, columns].astype(int).sort_index()
    if not labels.isin([0, 1]).all().all():
        raise ValueError(f"Non-binary labels in {path}")
    if (labels.sum(axis=1) == 0).any():
        raise ValueError(f"Cases without a root cause in {path}")
    return labels


def _read_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sample_index" not in frame:
        raise ValueError(f"Missing sample_index in {path}")
    forbidden = set(PROTOCOL["forbidden_features"]) - {"sample_index", "Unnamed: 0"}
    leaked = forbidden.intersection(frame.columns)
    # The preserved training table includes causes_type.  It is allowed only as
    # a quarantined column which is deleted before any feature discovery.
    leaked_without_quarantined = leaked - {"causes_type"}
    if leaked_without_quarantined:
        raise ValueError(f"Forbidden runtime columns in {path}: {sorted(leaked_without_quarantined)}")
    frame = frame.drop(columns=["Unnamed: 0", "causes_type"], errors="ignore")
    feature_columns = [column for column in frame.columns if column != "sample_index"]
    if not feature_columns:
        raise ValueError(f"No KPI columns in {path}")
    for column in feature_columns:
        frame[column] = frame[column].map(_numeric_cell).astype(float)
    return frame.sort_values("sample_index", kind="stable").reset_index(drop=True)


def _numeric_cell(value: object) -> float:
    if isinstance(value, str) and ";" in value:
        parts = pd.to_numeric(pd.Series(value.split(";")), errors="coerce")
        return float(parts.mean())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_labels(root: str | Path, split: str) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root)
    return _read_labels(root / f"{split}_label.csv")


def load_observations(root: str | Path, split: str) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root)
    return _read_observations(root / f"{split}_for_ml.csv")


def load_split(root: str | Path, split: str) -> SplitData:
    observations = load_observations(root, split)
    labels = load_labels(root, split)
    observed_ids = np.sort(observations["sample_index"].unique().astype(int))
    label_ids = labels.index.to_numpy(dtype=int)
    if not np.array_equal(observed_ids, label_ids):
        raise ValueError(
            f"Feature/label sample mismatch for {split}: "
            f"{len(observed_ids)} observed versus {len(label_ids)} labeled"
        )
    return SplitData(observations=observations, labels=labels, sample_ids=observed_ids)


def runtime_feature_columns(frame: pd.DataFrame) -> list[str]:
    forbidden = set(PROTOCOL["forbidden_features"])
    columns = [column for column in frame.columns if column not in forbidden]
    bad = forbidden.intersection(columns)
    if bad:
        raise AssertionError(f"Leakage guard failed: {sorted(bad)}")
    return columns

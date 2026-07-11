from __future__ import annotations

import numpy as np
import pandas as pd

from telco_mas.icas_spgc.dataset import runtime_feature_columns
from telco_mas.icas_spgc.features import build_feature_views
from telco_mas.icas_spgc.metrics import challenge_score, evaluate_predictions
from telco_mas.icas_spgc.models import apply_thresholds, calibrate_thresholds
from telco_mas.icas_spgc.protocol import PROTOCOL, protocol_hash


def test_official_challenge_score_penalizes_false_roots() -> None:
    labels = np.array([[1, 0, 0], [0, 1, 1]])
    predictions = np.array([[1, 1, 0], [0, 1, 0]])
    # case scores: 0/1 and 1/2
    assert challenge_score(predictions, labels) == 0.25
    assert evaluate_predictions(predictions, labels)["challenge_score"] == 0.25


def test_feature_views_never_contain_id_or_label_columns() -> None:
    frame = pd.DataFrame(
        {
            "sample_index": [7, 7, 9],
            "feature13": [1.0, 3.0, 8.0],
            "feature15": [2.0, 2.5, 3.0],
            "feature19": [-90.0, -91.0, -95.0],
            "feature20_distance": [1.0, 2.0, 3.0],
            "feature60": [-100.0, -99.0, -101.0],
            "featureX_if": [0.2, 0.3, 0.1],
            "featureY_if": [0.4, 0.5, 0.2],
            "feature0": [2.0, 4.0, 5.0],
        }
    )
    views = build_feature_views(frame)
    assert views.sample_ids.tolist() == [7, 9]
    assert "sample_index" not in runtime_feature_columns(frame)
    assert all("sample_index" not in column for column in views.full.columns)
    assert views.full.loc[7, "feature13__mean"] == 2.0
    assert len(views.level.columns) == 8 * 8
    assert len(views.dynamics.columns) == 8 * 6
    assert not any(column.startswith("feature0__") for column in views.causal.columns)
    assert all(column.startswith(("feature13__", "feature15__")) for column in views.causal_by_root[0])


def test_threshold_calibration_is_training_data_only_and_deterministic() -> None:
    probabilities = np.array([[0.9, 0.1, 0.2], [0.7, 0.8, 0.3], [0.2, 0.6, 0.9]])
    labels = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
    first = calibrate_thresholds(probabilities, labels)
    second = calibrate_thresholds(probabilities, labels)
    assert np.array_equal(first, second)
    prediction = apply_thresholds(probabilities, first)
    assert prediction.shape == labels.shape


def test_protocol_is_stable_and_resource_matched() -> None:
    assert len(protocol_hash()) == 64
    assert len(PROTOCOL["tree_seeds"]) == 3
    assert "causes_type" in PROTOCOL["forbidden_features"]
    assert PROTOCOL["systems"]["single"].startswith("three full-view")

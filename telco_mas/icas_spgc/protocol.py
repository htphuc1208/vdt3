"""Frozen confirmatory protocol for the ICASSP-SPGC 2022 experiment.

These constants are deliberately source controlled.  Changing any value creates
a new protocol version and invalidates comparisons with an earlier result file.
"""

from __future__ import annotations

import hashlib
import json


PROTOCOL = {
    "benchmark": "ICASSP-SPGC 2022 Root Cause Analysis for Wireless Network Fault Localization",
    "version": "icas-spgc-inductive-v3-final",
    "frozen_at": "2026-07-10",
    "feature_source": "variable-length train_for_ml.csv and test_for_ml.csv",
    "forbidden_features": [
        "causes_type",
        "root-cause(s)",
        "sample_index",
        "Unnamed: 0",
        "Root1",
        "Root2",
        "Root3",
        "Root4",
        "Root5",
        "Root6",
    ],
    "label_columns": ["Root1", "Root2", "Root3"],
    "validation_fraction": 0.30,
    "split_seed": 20260710,
    "oof_folds": 5,
    "tree_seeds": [101, 211, 307],
    "extra_trees": {
        "n_estimators": 400,
        "max_features": 0.7,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "threshold_grid": [round(x / 100, 2) for x in range(5, 96)],
    "bootstrap_repetitions": 10_000,
    "bootstrap_seed": 20260711,
    "published_reference_score": 0.93,
    "primary_metric": "official challenge score",
    "systems": {
        "single": "three full-view multi-label ExtraTrees bundles; mean probability",
        "multi": "level, dynamics, and causal-path multi-label ExtraTrees specialists; logistic adjudicator",
    },
}


def protocol_hash() -> str:
    payload = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

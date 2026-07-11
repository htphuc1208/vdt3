"""One-shot, preregistered single-versus-multi live-5G experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import load_split
from .features import FeatureViews, build_feature_views
from .metrics import evaluate_predictions, paired_comparison
from .models import (
    MultiAgent,
    SingleAgent,
    apply_thresholds,
    calibrate_thresholds,
    combo_strata,
)
from .protocol import PROTOCOL, protocol_hash
from .readiness import inspect


def _arrays(views: FeatureViews) -> tuple[np.ndarray, list[np.ndarray]]:
    return (
        views.full.to_numpy(dtype=float),
        [
            views.level.to_numpy(dtype=float),
            views.dynamics.to_numpy(dtype=float),
            views.causal.to_numpy(dtype=float),
        ],
    )


def _source_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run(root: str | Path) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(root)
    preflight = inspect(root, validate_test_labels=False)
    if not preflight["ready"]:
        raise RuntimeError(f"Dataset failed label-blind preflight checks: {preflight}")

    train = load_split(root, "train")
    train_views = build_feature_views(train.observations)
    if not np.array_equal(train.sample_ids, train_views.sample_ids):
        raise AssertionError("Training feature order mismatch")
    labels = train.labels.to_numpy(dtype=int)
    full, specialist_views = _arrays(train_views)

    all_indices = np.arange(len(labels))
    core_indices, validation_indices = train_test_split(
        all_indices,
        test_size=PROTOCOL["validation_fraction"],
        random_state=PROTOCOL["split_seed"],
        stratify=combo_strata(labels),
    )

    validation_started = time.perf_counter()
    single_core = SingleAgent.fit(full[core_indices], labels[core_indices])
    single_validation_probability = single_core.predict_proba(full[validation_indices])
    single_thresholds = calibrate_thresholds(
        single_validation_probability, labels[validation_indices]
    )
    single_validation_prediction = apply_thresholds(
        single_validation_probability, single_thresholds
    )

    # Meta-indices are local to the core arrays passed here.
    local_core = np.arange(len(core_indices))
    multi_core = MultiAgent.fit(
        [view[core_indices] for view in specialist_views],
        labels[core_indices],
        local_core,
    )
    multi_validation_probability = multi_core.predict_proba(
        [view[validation_indices] for view in specialist_views]
    )
    multi_thresholds = calibrate_thresholds(
        multi_validation_probability, labels[validation_indices]
    )
    multi_validation_prediction = apply_thresholds(multi_validation_probability, multi_thresholds)
    validation_seconds = time.perf_counter() - validation_started

    # The test labels are loaded only after every fitting and calibration choice
    # is complete.  No test-derived quantity enters either system.
    final_fit_started = time.perf_counter()
    single_final = SingleAgent.fit(full, labels)
    multi_final = MultiAgent.fit(specialist_views, labels, core_indices)
    fit_seconds = time.perf_counter() - final_fit_started

    test = load_split(root, "test")
    test_views = build_feature_views(test.observations)
    if not np.array_equal(test.sample_ids, test_views.sample_ids):
        raise AssertionError("Test feature order mismatch")
    test_full, test_specialist_views = _arrays(test_views)
    test_labels = test.labels.to_numpy(dtype=int)
    readiness = inspect(root, validate_test_labels=True)

    inference_started = time.perf_counter()
    single_test_probability = single_final.predict_proba(test_full)
    multi_test_probability = multi_final.predict_proba(test_specialist_views)
    inference_seconds = time.perf_counter() - inference_started
    single_test_prediction = apply_thresholds(single_test_probability, single_thresholds)
    multi_test_prediction = apply_thresholds(multi_test_probability, multi_thresholds)

    validation_metrics = {
        "single": evaluate_predictions(single_validation_prediction, labels[validation_indices]),
        "multi": evaluate_predictions(multi_validation_prediction, labels[validation_indices]),
        "paired": paired_comparison(
            multi_validation_prediction, single_validation_prediction, labels[validation_indices]
        ),
    }
    test_metrics = {
        "single": evaluate_predictions(single_test_prediction, test_labels),
        "multi": evaluate_predictions(multi_test_prediction, test_labels),
        "paired": paired_comparison(multi_test_prediction, single_test_prediction, test_labels),
    }
    reference = PROTOCOL["published_reference_score"]
    return {
        "protocol": PROTOCOL,
        "protocol_hash": protocol_hash(),
        "source_commit": _source_commit(root),
        "readiness": readiness,
        "label_blind_preflight": preflight,
        "data": {
            "train_cases": len(labels),
            "development_core_cases": len(core_indices),
            "validation_cases": len(validation_indices),
            "held_out_test_cases": len(test_labels),
            "full_features": full.shape[1],
            "specialist_features": {
                name: view.shape[1]
                for name, view in zip(("level", "dynamics", "causal"), specialist_views)
            },
        },
        "thresholds": {
            "single": single_thresholds.tolist(),
            "multi": multi_thresholds.tolist(),
            "selected_on": "training-only deterministic 30% validation split",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "published_reference": {
            "reported_third_place_score": reference,
            "multi_exceeds_reference": bool(test_metrics["multi"]["challenge_score"] > reference),
            "caveat": "The original challenge site is offline; comparison uses the preserved contestant test labels.",
        },
        "resource_match": {
            "single_base_model_bundles": 3,
            "multi_base_model_bundles": 3,
            "base_estimator_and_hyperparameters_identical": True,
            "multi_extra_component": "three per-root logistic adjudicators trained only on core OOF predictions",
        },
        "timing_seconds": {
            "validation_and_calibration": validation_seconds,
            "final_fit": fit_seconds,
            "test_inference_both_systems": inference_seconds,
            "total": time.perf_counter() - started,
        },
        "predictions": {
            "sample_index": test.sample_ids.tolist(),
            "single": single_test_prediction.tolist(),
            "multi": multi_test_prediction.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.root)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "protocol_hash": result["protocol_hash"],
        "validation": result["validation"],
        "test": result["test"],
        "published_reference": result["published_reference"],
        "out": str(path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

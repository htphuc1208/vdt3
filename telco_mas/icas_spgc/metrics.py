"""Official and secondary paired metrics for the wireless RCA challenge."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from .protocol import PROTOCOL


def case_scores(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=int)
    label = np.asarray(label, dtype=int)
    if prediction.shape != label.shape or prediction.ndim != 2:
        raise ValueError("prediction and label must be same-shaped 2D arrays")
    true_count = label.sum(axis=1)
    if np.any(true_count == 0):
        raise ValueError("every case must have at least one true root cause")
    plus = np.sum(prediction * label, axis=1)
    minus = np.sum(prediction * (1 - label), axis=1)
    return (plus - minus) / true_count


def challenge_score(prediction: np.ndarray, label: np.ndarray) -> float:
    return float(np.mean(case_scores(prediction, label)))


def evaluate_predictions(prediction: np.ndarray, label: np.ndarray) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=int)
    label = np.asarray(label, dtype=int)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        label, prediction, average=None, zero_division=0
    )
    return {
        "challenge_score": challenge_score(prediction, label),
        "micro_f1": float(f1_score(label, prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(label, prediction, average="macro", zero_division=0)),
        "exact_set_accuracy": float(accuracy_score(label, prediction)),
        "per_root": {
            root: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(per_f1[i]),
                "support": int(support[i]),
            }
            for i, root in enumerate(PROTOCOL["label_columns"])
        },
    }


def paired_comparison(
    multi_prediction: np.ndarray,
    single_prediction: np.ndarray,
    label: np.ndarray,
) -> dict[str, object]:
    multi_cases = case_scores(multi_prediction, label)
    single_cases = case_scores(single_prediction, label)
    delta = multi_cases - single_cases
    rng = np.random.default_rng(PROTOCOL["bootstrap_seed"])
    n = len(delta)
    boot = np.empty(PROTOCOL["bootstrap_repetitions"], dtype=float)
    for i in range(len(boot)):
        boot[i] = np.mean(delta[rng.integers(0, n, n)])
    multi_exact = np.all(np.asarray(multi_prediction) == label, axis=1)
    single_exact = np.all(np.asarray(single_prediction) == label, axis=1)
    multi_only = int(np.sum(multi_exact & ~single_exact))
    single_only = int(np.sum(single_exact & ~multi_exact))
    discordant = multi_only + single_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(multi_only, single_only), discordant, 0.5).pvalue
    )
    return {
        "mean_case_score_delta": float(np.mean(delta)),
        "paired_bootstrap_95_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "multi_better_cases": int(np.sum(delta > 0)),
        "single_better_cases": int(np.sum(delta < 0)),
        "ties": int(np.sum(delta == 0)),
        "mcnemar_exact": {
            "multi_only_correct": multi_only,
            "single_only_correct": single_only,
            "exact_binomial_p": p_value,
        },
    }

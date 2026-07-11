"""Resource-matched single-agent and specialist/adjudicator models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .protocol import PROTOCOL


def _tree(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(random_state=seed, **PROTOCOL["extra_trees"])


def _probabilities(model: ExtraTreesClassifier, values: np.ndarray) -> np.ndarray:
    output = []
    for probabilities, classes in zip(model.predict_proba(values), model.classes_):
        classes = np.asarray(classes)
        positive = np.flatnonzero(classes == 1)
        output.append(
            probabilities[:, positive[0]] if len(positive) else np.zeros(len(values), dtype=float)
        )
    return np.column_stack(output)


def combo_strata(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    powers = 1 << np.arange(labels.shape[1] - 1, -1, -1)
    return (labels * powers).sum(axis=1).astype(str)


@dataclass
class SingleAgent:
    models: list[ExtraTreesClassifier]

    @classmethod
    def fit(cls, full: np.ndarray, labels: np.ndarray) -> "SingleAgent":
        models = [_tree(seed).fit(full, labels) for seed in PROTOCOL["tree_seeds"]]
        return cls(models=models)

    def predict_proba(self, full: np.ndarray) -> np.ndarray:
        return np.mean([_probabilities(model, full) for model in self.models], axis=0)


def _adjudicator_input(specialist_probabilities: np.ndarray) -> np.ndarray:
    # [cases, specialists, roots] -> one small evidence vector per root.
    mean = specialist_probabilities.mean(axis=1, keepdims=True)
    spread = specialist_probabilities.std(axis=1, keepdims=True)
    minimum = specialist_probabilities.min(axis=1, keepdims=True)
    maximum = specialist_probabilities.max(axis=1, keepdims=True)
    return np.concatenate([specialist_probabilities, mean, spread, minimum, maximum], axis=1)


@dataclass
class MultiAgent:
    specialists: list[ExtraTreesClassifier]
    adjudicators: list[LogisticRegression]

    @classmethod
    def fit(
        cls,
        views: list[np.ndarray],
        labels: np.ndarray,
        meta_indices: np.ndarray,
    ) -> "MultiAgent":
        labels = np.asarray(labels, dtype=int)
        strata = combo_strata(labels[meta_indices])
        oof = np.zeros((len(meta_indices), len(views), labels.shape[1]), dtype=float)
        folds = StratifiedKFold(
            n_splits=PROTOCOL["oof_folds"], shuffle=True, random_state=PROTOCOL["split_seed"]
        )
        for train_local, held_local in folds.split(meta_indices, strata):
            train_indices = meta_indices[train_local]
            held_indices = meta_indices[held_local]
            for specialist_index, (view, seed) in enumerate(zip(views, PROTOCOL["tree_seeds"])):
                model = _tree(seed).fit(view[train_indices], labels[train_indices])
                oof[held_local, specialist_index] = _probabilities(model, view[held_indices])
        evidence = _adjudicator_input(oof)
        adjudicators = []
        for root in range(labels.shape[1]):
            adjudicator = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=PROTOCOL["split_seed"] + root,
            )
            adjudicator.fit(evidence[:, :, root], labels[meta_indices, root])
            adjudicators.append(adjudicator)
        specialists = [
            _tree(seed).fit(view, labels)
            for view, seed in zip(views, PROTOCOL["tree_seeds"])
        ]
        return cls(specialists=specialists, adjudicators=adjudicators)

    def predict_proba(self, views: list[np.ndarray]) -> np.ndarray:
        specialist_probabilities = np.stack(
            [_probabilities(model, view) for model, view in zip(self.specialists, views)], axis=1
        )
        evidence = _adjudicator_input(specialist_probabilities)
        return np.column_stack(
            [
                adjudicator.predict_proba(evidence[:, :, root])[:, 1]
                for root, adjudicator in enumerate(self.adjudicators)
            ]
        )


def calibrate_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    true_count = labels.sum(axis=1)
    thresholds: list[float] = []
    for root in range(labels.shape[1]):
        candidates = []
        signed_value = (2 * labels[:, root] - 1) / true_count
        for threshold in PROTOCOL["threshold_grid"]:
            contribution = np.mean((probabilities[:, root] >= threshold) * signed_value)
            candidates.append((float(contribution), -abs(threshold - 0.5), threshold))
        thresholds.append(float(max(candidates)[2]))
    return np.asarray(thresholds)


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (np.asarray(probabilities) >= np.asarray(thresholds)[None, :]).astype(int)

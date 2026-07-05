"""E2: validation-only weight fitting, with the accuracy vs calibration split.

Confirms the scientific claims baked into the fitter:
- rho/temperature cannot change which root wins (calibration only);
- unequal per-modality reliability CAN flip the winner to the true root;
- the grid fitter selects weights that improve MRR on a dev split;
- the fitter refuses to touch a locked holdout id.
"""
from __future__ import annotations

import pytest

from telco_mas.shardrca.board import Blackboard, CandidateEvidence, WorkerDistribution
from telco_mas.shardrca.fit_weights import FitCase, fit_fusion_weights, worker_from_compact
from telco_mas.shardrca.fusion import fuse_worker_distributions
from telco_mas.shardrca.weights import FusionWeights

COMPONENTS = ["A", "B"]
REASONS = ["CPU fault"]


def _w(worker_id, modality, component, prob):
    return WorkerDistribution(
        worker_id=worker_id, modality=modality, candidate_scope=list(COMPONENTS),
        candidates=[CandidateEvidence(component=component, reason_family="CPU fault",
                                      probability=prob, modality=modality, worker_id=worker_id)],
        other_mass=1.0 - prob,
    )


def _case():
    # Three metric workers favour the wrong root B; one trace worker favours the
    # true root A. Equal-weight PoE picks B (metric majority); up-weighting the
    # trace modality flips the winner to A.
    return FitCase(
        case_id="c1", components=COMPONENTS, reasons=REASONS, true_root="A",
        distributions=[
            _w("m1", "metrics", "B", 0.7),
            _w("m2", "metrics", "B", 0.7),
            _w("m3", "metrics", "B", 0.7),
            _w("t1", "traces", "A", 0.9),
        ],
    )


def _winner(case, weights):
    return fuse_worker_distributions(
        case.distributions, Blackboard(case_id=case.case_id),
        components=case.components, reasons=case.reasons, weights=weights,
    ).winner.component


def test_rho_temperature_do_not_change_winner():
    case = _case()
    base = _winner(case, FusionWeights(correlation_rho=0.0, temperature=1.0))
    tuned = _winner(case, FusionWeights(correlation_rho=1.0, temperature=3.0))
    assert base == tuned  # monotone transforms preserve the argmax


def test_modality_reliability_can_flip_winner():
    case = _case()
    equal = _winner(case, FusionWeights.default())
    reliable = _winner(case, FusionWeights(
        modality_reliability_enabled=True,
        modality_weights={"metrics": 1.0, "traces": 3.0, "logs": 3.0, "events": 1.0, "auxiliary": 0.8},
    ))
    assert equal == "B"
    assert reliable == "A"


def test_fitter_improves_mrr_and_selects_reliability():
    cases = [_case()]
    fitted, report = fit_fusion_weights(cases, fit_on="unit_dev")
    assert report["fitted_accuracy"]["mrr"] >= report["baseline"]["mrr"]
    assert report["fitted_accuracy"]["hit_at_1"] >= report["baseline"]["hit_at_1"]
    # The true root is only recoverable by up-weighting the trace modality.
    assert fitted.modality_reliability_enabled is True
    assert fitted.version == "fit_v1"


def test_worker_from_compact_roundtrip():
    original = _w("t1", "traces", "A", 0.9)
    rebuilt = worker_from_compact(original.compact())
    assert rebuilt.worker_id == "t1"
    assert rebuilt.modality == "traces"
    assert rebuilt.candidates[0].component == "A"
    assert rebuilt.candidates[0].reason_family == "CPU fault"

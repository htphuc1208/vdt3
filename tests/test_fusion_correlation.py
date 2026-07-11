"""B: optional correlation discount must not treat agreeing workers as independent.

Equal-weight product-of-experts multiplies every worker's likelihood, so N
workers that all flag the *same* propagated root (as on the real telecom path,
where every worker shares the full candidate scope) drive the fused posterior
toward certainty even though they carry roughly one source's worth of
independent information. The optional effective-number-of-experts discount reads
positive Pearson correlation from component posterior vectors, shrinks that
over-confidence when workers agree, and leaves genuinely disagreeing workers at
full weight.
"""
from __future__ import annotations

from telco_mas.shardrca.board import Blackboard, CandidateEvidence, WorkerDistribution
from telco_mas.shardrca.fusion import _worker_effective_weight, fuse_worker_distributions
from telco_mas.shardrca.weights import FusionWeights

COMPONENTS = ["A", "B", "C", "D"]
REASONS = ["CPU fault", "network delay"]


def _worker(worker_id: str, component: str, modality: str, prob: float = 0.7) -> WorkerDistribution:
    # Full candidate scope, mirroring the real OpenRCA path (all TELECOM_COMPONENTS).
    return WorkerDistribution(
        worker_id=worker_id,
        modality=modality,
        candidate_scope=list(COMPONENTS),
        candidates=[
            CandidateEvidence(
                component=component, reason_family="CPU fault", probability=prob,
                modality=modality, worker_id=worker_id,
            )
        ],
        other_mass=0.3,
    )


def _agreeing_workers(n: int) -> list[WorkerDistribution]:
    modalities = ["metrics", "logs", "traces", "events"]
    return [_worker(f"w{i}", "A", modalities[i % len(modalities)]) for i in range(n)]


def _fuse(workers, weights):
    return fuse_worker_distributions(
        workers, Blackboard(case_id="t"), components=COMPONENTS, reasons=REASONS, weights=weights
    )


def test_default_reduces_to_equal_weight_poe():
    workers = _agreeing_workers(3)
    default = _fuse(workers, FusionWeights.default())
    explicit = _fuse(workers, FusionWeights(correlation_rho=0.0, temperature=1.0))
    assert default.winner.component == "A"
    assert abs(default.winner.confidence - explicit.winner.confidence) < 1e-9


def test_agreeing_workers_are_discounted():
    workers = _agreeing_workers(4)
    poe = _fuse(workers, FusionWeights(correlation_rho=0.0))
    aware = _fuse(workers, FusionWeights(correlation_rho=1.0))
    assert poe.winner.component == aware.winner.component == "A"
    # Four fully-agreeing workers must yield less certainty once redundancy is modelled.
    assert aware.winner.confidence < poe.winner.confidence


def test_effective_weight_full_agreement_vs_disagreement():
    agree = [_worker("w0", "A", "metrics"), _worker("w1", "A", "logs"), _worker("w2", "A", "traces")]
    disagree = [_worker("w0", "A", "metrics"), _worker("w1", "B", "logs"), _worker("w2", "C", "traces")]
    w_agree = _worker_effective_weight(agree, set(COMPONENTS), correlation_rho=1.0)
    w_disagree = _worker_effective_weight(disagree, set(COMPONENTS), correlation_rho=1.0)
    # Full agreement collapses toward one effective expert (weight ~1/N);
    # full disagreement keeps them independent (weight ~1).
    assert w_agree < 0.5
    assert w_disagree == 1.0
    assert w_agree < w_disagree


def test_rho_zero_is_full_independence():
    workers = _agreeing_workers(3)
    assert _worker_effective_weight(workers, set(COMPONENTS), correlation_rho=0.0) == 1.0


def test_temperature_flattens_posterior():
    workers = _agreeing_workers(3)
    sharp = _fuse(workers, FusionWeights(temperature=1.0))
    tempered = _fuse(workers, FusionWeights(temperature=3.0))
    assert tempered.winner.confidence < sharp.winner.confidence


def test_weights_freeze_and_reload(tmp_path):
    w = FusionWeights(version="fit_v1", correlation_rho=0.6, temperature=1.4, fit_on="rcaeval_hard_dev")
    path = w.freeze(tmp_path / "weights.json")
    reloaded = FusionWeights.load(path)
    assert reloaded.correlation_rho == 0.6
    assert reloaded.temperature == 1.4
    assert reloaded.version == "fit_v1"

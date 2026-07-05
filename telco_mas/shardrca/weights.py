"""Versioned, freezable fusion/consensus weights.

Rationale (addresses the reproducibility critique): the fusion and convergence
constants used to be hard-coded magic numbers scattered across ``fusion.py``.
That makes it impossible to (a) fit them on a validation split, (b) *freeze*
them before a confirmatory run, or (c) tell a reviewer which numbers produced a
result. This module turns them into a single artifact that can be loaded from a
JSON file (``SHARDRCA_WEIGHTS`` env var or an explicit path), carries provenance,
and defaults to the historical constants so nothing changes until a fit is frozen.

The default is a strict no-op: ``redundancy_lambda=0`` and ``temperature=1``
reduce the correlation-aware pool exactly to the previous equal-weight
product-of-experts, so existing behaviour/tests are unchanged until a validation
fit is deliberately loaded.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_ENV_VAR = "SHARDRCA_WEIGHTS"

# Historical constants preserved verbatim from fusion.py.
_DEFAULT_MODALITY_WEIGHTS: dict[str, float] = {
    "metrics": 1.0,
    "logs": 1.15,
    "traces": 1.15,
    "events": 1.05,
    "auxiliary": 0.8,
}


@dataclass(frozen=True)
class FusionWeights:
    """All tunable fusion/convergence constants in one freezable place.

    ``redundancy_lambda`` discounts workers whose candidate scope overlaps other
    workers (correlated evidence must not be multiplied as if independent).
    ``temperature`` > 1 flattens the fused posterior to fight product-of-experts
    overconfidence; the default 1.0 is a no-op.
    """

    version: str = "default_v0"
    fit_on: str = "none (historical constants, not fitted)"
    provenance: str = ""
    # Correlation-aware log-opinion-pool controls (B).
    # ``correlation_rho`` in [0, 1] is how much observed cross-worker *agreement*
    # is attributed to redundancy rather than independent confirmation. 0 keeps
    # the equal-weight product of experts (N independent experts); 1 collapses a
    # fully-agreeing ensemble to one effective expert (weighted geometric mean).
    correlation_rho: float = 0.0
    temperature: float = 1.0
    # Per-worker modality reliability. rho/temperature are monotone transforms of
    # the fused posterior and cannot change the argmax (hence cannot move Hit@1);
    # they only recalibrate confidence. Unequal per-modality weights DO change the
    # ranking, which is the mechanism that can overturn the same-board oracle.
    # Off by default so the frozen equal-weight OpenRCA result stays reproducible;
    # a validation-fitted artifact turns it on.
    modality_reliability_enabled: bool = False
    # Topology-aware re-ranking strength: score *= (1 + gamma * explanatory_coverage)
    # over the dependency graph. 0 = no-op; > 0 lets an upstream cause outrank a
    # downstream symptom (a ranking-changing lever, like modality reliability).
    topology_gamma: float = 0.0
    # Temporal-precedence re-ranking strength: score *= (1 + beta * precedence),
    # where precedence favours the earliest-onset component (a root precedes its
    # symptoms). 0 = no-op; also a ranking-changing lever.
    temporal_beta: float = 0.0
    # Candidate-evidence convergence bonuses (fuse_candidate_evidence).
    convergence_modality_bonus: float = 0.18
    convergence_shard_bonus: float = 0.05
    modality_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_MODALITY_WEIGHTS))

    def modality_weight(self, modality: str) -> float:
        return self.modality_weights.get(str(modality), 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def default(cls) -> "FusionWeights":
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "FusionWeights":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in allowed}
        mw = payload.get("modality_weights")
        if isinstance(mw, dict):
            payload["modality_weights"] = {str(k): float(v) for k, v in mw.items()}
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FusionWeights":
        """Load frozen weights from ``path``, else ``$SHARDRCA_WEIGHTS``, else default."""
        target = path or os.environ.get(_ENV_VAR)
        if not target:
            return cls.default()
        p = Path(target)
        if not p.exists():
            return cls.default()
        return cls.from_dict(json.loads(p.read_text()))

    def freeze(self, path: str | Path) -> Path:
        """Persist these weights as a frozen JSON artifact and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return p

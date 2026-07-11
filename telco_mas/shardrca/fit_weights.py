"""Fit and freeze ShardRCA fusion weights on a validation split.

Scientific contract (PLAN v6 guardrails): weights are fitted **only** on declared
development/validation data and then frozen as exploratory artifacts. Current
confirmatory runners use default no-fit weights and reject ``SHARDRCA_WEIGHTS``
unless a future preregistration explicitly permits a validation-frozen artifact.
This module never reads a locked holdout -- pass its case IDs via ``--forbid`` and
the fitter aborts if any appear.

What actually moves accuracy: ``correlation_rho`` and ``temperature`` are monotone
transforms of the fused posterior and cannot change the argmax, so they are tuned
for calibration (Brier score) only. Unequal per-modality reliability weights can
change the ranking, so they are tuned to maximise mean reciprocal rank of the true
root. The search is a small, declared grid to keep the fit auditable.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from .board import CandidateEvidence, WorkerDistribution
from .fusion import fuse_worker_distributions
from .weights import FusionWeights


@dataclass
class FitCase:
    case_id: str
    components: list[str]
    reasons: list[str]
    true_root: str
    distributions: list[WorkerDistribution]


def worker_from_compact(data: dict[str, Any]) -> WorkerDistribution:
    """Rebuild a WorkerDistribution from a runner artifact's compact() dict."""
    candidates = []
    for c in data.get("candidates", []):
        candidates.append(
            CandidateEvidence(
                component=c.get("component", ""),
                reason_family=c.get("reason_family", "unknown"),
                support_score=float(c.get("support_score", 0.0)),
                refute_score=float(c.get("refute_score", 0.0)),
                probability=float(c.get("probability", 0.0)),
                modality=c.get("modality", "auxiliary"),
                worker_id=c.get("worker_id", data.get("worker_id", "")),
                shard_id=c.get("shard_id", ""),
                evidence_ptrs=list(c.get("evidence", []) or []),
                missing_evidence=list(c.get("missing_evidence", []) or []),
                local_rank=int(c.get("local_rank", 1)),
                rationale=c.get("rationale", ""),
            )
        )
    return WorkerDistribution(
        worker_id=data.get("worker_id", ""),
        modality=data.get("modality", "auxiliary"),
        candidate_scope=list(data.get("candidate_scope", [])),
        candidates=candidates,
        other_mass=float(data.get("other_mass", 0.0)),
        notes=data.get("notes", ""),
    )


def _reciprocal_rank(candidates, true_root: str) -> float:
    needle = (true_root or "").strip().lower()
    for idx, cand in enumerate(candidates, start=1):
        if cand.component.strip().lower() == needle:
            return 1.0 / idx
    return 0.0


def _score(cases: list[FitCase], weights: FusionWeights) -> dict[str, float]:
    mrr = 0.0
    hit1 = 0.0
    brier = 0.0
    for case in cases:
        result = fuse_worker_distributions(
            case.distributions, _EmptyBoard(case.case_id),
            components=case.components, reasons=case.reasons, weights=weights,
        )
        rr = _reciprocal_rank(result.candidates, case.true_root)
        mrr += rr
        hit1 += 1.0 if rr == 1.0 else 0.0
        correct = 1.0 if result.winner.component.strip().lower() == case.true_root.strip().lower() else 0.0
        brier += (result.winner.confidence - correct) ** 2
    n = max(1, len(cases))
    return {"mrr": mrr / n, "hit_at_1": hit1 / n, "brier": brier / n}


class _EmptyBoard:
    """Minimal board stand-in: fusion only queries evidence_for() for timestamps."""

    def __init__(self, case_id: str):
        self.case_id = case_id

    def evidence_for(self, *_args, **_kwargs):
        return []


def fit_fusion_weights(
    cases: list[FitCase],
    *,
    modality_boost_grid: tuple[float, ...] = (1.0, 1.15, 1.3, 1.5, 2.0),
    rho_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    temperature_grid: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    fit_on: str = "unspecified_validation_split",
    version: str = "fit_v1",
) -> tuple[FusionWeights, dict]:
    """Grid-fit reliability weights (for accuracy) then rho/temperature (for calibration)."""

    baseline = _score(cases, FusionWeights.default())

    # Stage 1 — reliability weights maximise MRR (they change the ranking).
    best_acc = baseline
    best_boost = 1.0
    best_enabled = False
    for boost in modality_boost_grid:
        weights = FusionWeights(
            modality_reliability_enabled=boost != 1.0,
            modality_weights={"metrics": 1.0, "logs": boost, "traces": boost, "events": 1.0 + (boost - 1.0) / 2, "auxiliary": 0.8},
        )
        score = _score(cases, weights)
        if (score["mrr"], score["hit_at_1"]) > (best_acc["mrr"], best_acc["hit_at_1"]):
            best_acc, best_boost, best_enabled = score, boost, boost != 1.0

    # Stage 2 — rho/temperature minimise Brier (calibration) at the fixed ranking.
    best_cal = None
    best_rho, best_temp = 0.0, 1.0
    for rho, temp in product(rho_grid, temperature_grid):
        weights = FusionWeights(
            correlation_rho=rho, temperature=temp,
            modality_reliability_enabled=best_enabled,
            modality_weights={"metrics": 1.0, "logs": best_boost, "traces": best_boost, "events": 1.0 + (best_boost - 1.0) / 2, "auxiliary": 0.8},
        )
        score = _score(cases, weights)
        if best_cal is None or score["brier"] < best_cal["brier"]:
            best_cal, best_rho, best_temp = score, rho, temp

    fitted = FusionWeights(
        version=version,
        fit_on=fit_on,
        provenance=f"grid fit over {len(cases)} validation cases",
        correlation_rho=best_rho,
        temperature=best_temp,
        modality_reliability_enabled=best_enabled,
        modality_weights={"metrics": 1.0, "logs": best_boost, "traces": best_boost, "events": 1.0 + (best_boost - 1.0) / 2, "auxiliary": 0.8},
    )
    report = {
        "n_cases": len(cases),
        "baseline": baseline,
        "fitted_accuracy": best_acc,
        "fitted_calibration": best_cal,
        "selected": {"modality_boost": best_boost, "correlation_rho": best_rho, "temperature": best_temp},
    }
    return fitted, report


def _is_compact(dist: dict[str, Any]) -> bool:
    """Runner artifacts serialize candidate evidence pointers under 'evidence'."""
    candidates = dist.get("candidates") or []
    return bool(candidates) and "evidence" in candidates[0] and "evidence_ptrs" not in candidates[0]


def load_cases(path: str | Path, *, forbid: set[str] | None = None) -> list[FitCase]:
    data = json.loads(Path(path).read_text())
    forbid = forbid or set()
    cases: list[FitCase] = []
    for row in data.get("cases", data if isinstance(data, list) else []):
        case_id = str(row.get("case_id"))
        if case_id in forbid:
            raise ValueError(f"case {case_id} is a locked holdout; refusing to fit on it")
        dists = [
            # Runner artifacts store the compact() form (candidate "evidence" key);
            # a hand-written dev file may use the full WorkerDistribution schema.
            worker_from_compact(d) if _is_compact(d) else WorkerDistribution.model_validate(d)
            for d in row.get("distributions", [])
        ]
        cases.append(FitCase(
            case_id=case_id,
            components=list(row["components"]),
            reasons=list(row["reasons"]),
            true_root=str(row["true_root"]),
            distributions=dists,
        ))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit and freeze ShardRCA fusion weights on validation data")
    parser.add_argument("--dev", required=True, help="validation cases JSON (never a locked holdout)")
    parser.add_argument("--out", required=True, help="frozen FusionWeights artifact path")
    parser.add_argument("--forbid", nargs="*", default=[], help="locked holdout case IDs to refuse")
    parser.add_argument("--fit-on", default="unspecified_validation_split")
    parser.add_argument("--version", default="fit_v1")
    args = parser.parse_args(argv)

    cases = load_cases(args.dev, forbid=set(args.forbid))
    fitted, report = fit_fusion_weights(cases, fit_on=args.fit_on, version=args.version)
    fitted.freeze(args.out)
    print(json.dumps({"frozen": args.out, "weights": fitted.to_dict(), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

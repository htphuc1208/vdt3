"""Validation-only fitter for OpenRCA ShardRCA repair weights.

The default claim protocol does not consume this output. A tuned repair-weight
artifact can support evidence only after a separate preregistration names the
validation rows, forbidden holdout rows, artifact path, and algorithm ID before
the holdout is run.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from ..shardrca.board import Blackboard, Finding, WorkerDistribution
from ..shardrca.fit_weights import worker_from_compact
from ..shardrca.fusion import fuse_worker_distributions
from ..shardrca.temporal import temporal_rerank
from ..shardrca.topology import DependencyGraph, topology_rerank
from ..shardrca.weights import FusionWeights
from .tools import TELECOM_COMPONENTS, TELECOM_REASONS


@dataclass
class OpenRCAFitCase:
    row_id: str
    true_component: str
    components: list[str]
    reasons: list[str]
    distributions: list[WorkerDistribution]
    board: Blackboard
    graph: DependencyGraph | None
    symptomatic: set[str]


def fit_openrca_repair_weights(
    cases: list[OpenRCAFitCase],
    *,
    version: str = "openrca_repair_v1",
    fit_on: str = "declared_openrca_validation_rows",
) -> tuple[FusionWeights, dict[str, Any]]:
    baseline = _score(cases, FusionWeights.default())
    best_weights = FusionWeights.default()
    best_score = baseline
    trials = []
    for boost, rho, temp, gamma, beta in product(
        (1.0, 1.2, 1.5, 2.0),
        (0.0, 0.5, 1.0),
        (1.0, 2.0),
        (0.0, 1.0, 2.0),
        (0.0, 1.0, 2.0),
    ):
        weights = FusionWeights(
            version=version,
            fit_on=fit_on,
            provenance=f"grid fit over {len(cases)} declared OpenRCA validation rows",
            correlation_rho=rho,
            temperature=temp,
            modality_reliability_enabled=boost != 1.0,
            topology_gamma=gamma,
            temporal_beta=beta,
            modality_weights={
                "metrics": 1.0,
                "logs": boost,
                "traces": boost,
                "events": 1.0 + (boost - 1.0) / 2,
                "auxiliary": 0.8,
            },
        )
        score = _score(cases, weights)
        trials.append({"boost": boost, "rho": rho, "temperature": temp, "gamma": gamma, "beta": beta, **score})
        if (score["hit_at_1"], score["mrr"], -score["brier"]) > (
            best_score["hit_at_1"],
            best_score["mrr"],
            -best_score["brier"],
        ):
            best_score = score
            best_weights = weights
    report = {
        "n_cases": len(cases),
        "baseline": baseline,
        "fitted": best_score,
        "selected": best_weights.to_dict(),
        "trials_top": sorted(trials, key=lambda item: (-item["hit_at_1"], -item["mrr"], item["brier"]))[:12],
        "contract": "validation-only; forbidden prereg rows are refused before fitting",
    }
    return best_weights, report


def load_cases(
    paths: list[str | Path],
    *,
    dev_rows: set[str],
    forbidden_rows: set[str],
) -> list[OpenRCAFitCase]:
    if not dev_rows:
        raise ValueError("--dev-rows is required; fitter will not infer validation rows")
    overlap = dev_rows & forbidden_rows
    if overlap:
        raise ValueError(f"declared dev rows overlap forbidden confirmatory rows: {sorted(overlap)}")
    cases: list[OpenRCAFitCase] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            row_id = str(row.get("row_id"))
            if row_id not in dev_rows:
                continue
            if row_id in forbidden_rows:
                raise ValueError(f"row {row_id} is forbidden by preregistration")
            if str(row.get("system")) != "shardrca_full":
                continue
            artifacts = row.get("artifacts") if isinstance(row.get("artifacts"), dict) else {}
            distributions = [
                worker_from_compact(item)
                for item in artifacts.get("worker_distributions", []) or []
                if isinstance(item, dict)
            ]
            if not distributions:
                continue
            true_component = _expected_component(str(row.get("scoring_points") or ""))
            if not true_component:
                continue
            catalog = artifacts.get("candidate_catalog") if isinstance(artifacts.get("candidate_catalog"), dict) else {}
            board = _board_from_artifact(row_id, artifacts)
            topology = artifacts.get("topology") if isinstance(artifacts.get("topology"), dict) else {}
            graph = DependencyGraph.from_edges(topology.get("edges") or []) if topology.get("edges") else None
            symptomatic = set(topology.get("symptomatic_components") or [component for component, _ in board.top_components(20)])
            cases.append(
                OpenRCAFitCase(
                    row_id=row_id,
                    true_component=true_component,
                    components=list(catalog.get("components") or TELECOM_COMPONENTS),
                    reasons=list(catalog.get("reasons") or TELECOM_REASONS),
                    distributions=distributions,
                    board=board,
                    graph=graph,
                    symptomatic=symptomatic,
                )
            )
    return cases


def forbidden_rows_from_preregs(paths: list[str | Path]) -> set[str]:
    forbidden: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        row_selection = payload.get("row_selection", {})
        forbidden.update(str(item) for item in row_selection.get("row_ids", []) or [])
        forbidden.update(str(item) for item in row_selection.get("confirmatory_row_ids", []) or [])
    return forbidden


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit OpenRCA repair weights on declared validation rows only")
    parser.add_argument("paths", nargs="+", help="OpenRCA result artifacts containing shardrca_full worker artifacts")
    parser.add_argument("--dev-rows", required=True, help="comma-separated validation row IDs permitted for fitting")
    parser.add_argument("--forbid-prereg", action="append", default=[], help="frozen prereg JSON whose rows must be refused")
    parser.add_argument("--out-weights", default="results/weights/validation_openrca_repair_v1.json")
    parser.add_argument("--out-report", default="results/validation_openrca_repair_fit_report.json")
    parser.add_argument("--version", default="openrca_repair_v1")
    parser.add_argument("--fit-on", default="declared_openrca_validation_rows")
    args = parser.parse_args(argv)
    dev_rows = {item.strip() for item in args.dev_rows.split(",") if item.strip()}
    forbidden = forbidden_rows_from_preregs(args.forbid_prereg)
    cases = load_cases(args.paths, dev_rows=dev_rows, forbidden_rows=forbidden)
    weights, report = fit_openrca_repair_weights(cases, version=args.version, fit_on=args.fit_on)
    weights.freeze(args.out_weights)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps({"weights": weights.to_dict(), "report": report}, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_weights": args.out_weights,
        "out_report": args.out_report,
        "n_cases": len(cases),
        "baseline": report["baseline"],
        "fitted": report["fitted"],
    }, indent=2))
    return 0


def _score(cases: list[OpenRCAFitCase], weights: FusionWeights) -> dict[str, float]:
    hit = 0.0
    mrr = 0.0
    brier = 0.0
    for case in cases:
        result = fuse_worker_distributions(
            case.distributions,
            case.board,
            components=case.components,
            reasons=case.reasons,
            weights=weights,
        )
        result = topology_rerank(result, case.symptomatic, case.graph, gamma=weights.topology_gamma)
        result = temporal_rerank(result, case.board, beta=weights.temporal_beta)
        ranked = [candidate.component for candidate in result.candidates]
        rr = _reciprocal_rank(ranked, case.true_component)
        mrr += rr
        hit += 1.0 if rr == 1.0 else 0.0
        correct = 1.0 if result.winner.component == case.true_component else 0.0
        brier += (result.winner.confidence - correct) ** 2
    n = max(1, len(cases))
    return {"hit_at_1": round(hit / n, 4), "mrr": round(mrr / n, 4), "brier": round(brier / n, 6)}


def _reciprocal_rank(ranked: list[str], true_component: str) -> float:
    for index, component in enumerate(ranked, start=1):
        if component == true_component:
            return 1.0 / index
    return 0.0


def _expected_component(scoring_points: str) -> str:
    match = re.search(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)", scoring_points)
    return match.group(1).strip() if match else ""


def _board_from_artifact(row_id: str, artifacts: dict[str, Any]) -> Blackboard:
    findings = []
    for item in artifacts.get("board_findings", []) or []:
        if not isinstance(item, dict):
            continue
        window = item.get("window") if isinstance(item.get("window"), list) else [None, None]
        findings.append(
            Finding(
                shard_id="artifact",
                modality=item.get("modality", "auxiliary"),
                component=str(item.get("component") or ""),
                signal=str(item.get("signal") or ""),
                direction=str(item.get("direction") or ""),
                magnitude=float(item.get("magnitude") or 0.0),
                score=float(item.get("score") or 0.0),
                window_start=window[0] if window else None,
                window_end=window[1] if len(window) > 1 else None,
                evidence_ptr=str(item.get("evidence") or ""),
                summary=str(item.get("summary") or ""),
            )
        )
    return Blackboard(case_id=row_id, findings=findings)


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline (no-LLM) fit of local-fusion weights on a disjoint RCAEval-hard dev split.

This exercises the full fit -> freeze -> load pipeline without spending LLM budget
and without touching any locked/confirmatory data. It fits the weights that affect
the deterministic ``fuse_candidate_evidence`` path (modality reliability + modality
convergence bonuses), which — unlike ``correlation_rho``/``temperature`` — actually
change the ranking and therefore Hit@1.

Scientific status (honest):
- The offline local-fusion path is a *proxy* for the live LLM ``shardrca_full``
  synthesizer; these weights are a validated prior, not a confirmatory result.
- The dev/val cases are selected disjoint from the locked v7 holdout AND the C
  high-volume confirmatory split, so nothing here contaminates a future claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..evaluation.stats import accuracy_at_k, reciprocal_rank
from .board import Blackboard
from .catalog import build_catalog_for_case, make_component_group_shards
from .fusion import candidate_evidence_from_findings, fuse_candidate_evidence
from .hard_split import load_hard_cases
from .miner import MinerWorker
from .temporal import temporal_rerank
from .topology import DependencyGraph, build_service_graph_from_trace_rows, topology_rerank
from .weights import FusionWeights


def runtime_case_id(raw_case_id: str) -> str:
    """Reproduce hard_split._runtime_case_id from an adapter raw case id.

    The adapter raw id ``RCAEval-<source>-<fault>-<leaf>`` is exactly the string
    hashed by the split, so ``<source>-<sha256[:12]>`` matches the split ids.
    """
    parts = raw_case_id.split("-")
    source = f"{parts[1]}-{parts[2]}" if len(parts) >= 3 else raw_case_id
    digest = hashlib.sha256(raw_case_id.encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}"


@dataclass
class LocalFusionCase:
    case_id: str
    runtime_id: str
    true_root: str
    evidence: list
    board: Blackboard
    graph: DependencyGraph | None = None
    symptomatic: set = None  # type: ignore[assignment]


def _trace_graph(catalog, *, max_rows: int = 200_000) -> DependencyGraph | None:
    """Build a service dependency graph from the case's raw traces.csv, if present.

    Only the span/parent/service columns and a bounded number of rows are read:
    the service call graph is a small, low-cardinality structure that a sample of
    spans recovers, so we avoid loading multi-hundred-MB trace files in full.
    """
    trace_files = [f.path for f in catalog.files if f.relative_path.lower() == "traces.csv"]
    if not trace_files:
        return None
    try:
        import pandas as pd
        frame = pd.read_csv(
            trace_files[0],
            usecols=lambda c: c in {"spanID", "parentSpanID", "serviceName"},
            nrows=max_rows,
        )
    except Exception:
        return None
    if frame.empty or "serviceName" not in frame:
        return None
    return build_service_graph_from_trace_rows(frame.to_dict("records"))


def _has_trace_file(case) -> bool:
    """Cheap check (catalog file listing only, no mining) for a usable trace graph."""
    try:
        catalog = build_catalog_for_case(case, compute_ranges=False)
    except Exception:
        return False
    return any(f.relative_path.lower() == "traces.csv" for f in catalog.files)


def extract_cases(cases, *, finding_limit: int = 25, chunksize: int = 50_000) -> list[LocalFusionCase]:
    """Run the deterministic offline miners once per case and cache local evidence + graph."""
    out: list[LocalFusionCase] = []
    for case in cases:
        catalog = build_catalog_for_case(case, compute_ranges=False)
        shards = make_component_group_shards(catalog, group_size=6)
        board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
        worker = MinerWorker(limit=finding_limit, chunksize=chunksize)
        evidence = []
        for result in worker.run_many(shards):
            board.extend(result.findings)
            evidence.extend(candidate_evidence_from_findings(result.shard_id, result.findings))
        symptomatic = {component for component, score in board.top_components(20) if score > 0}
        out.append(LocalFusionCase(
            case_id=case.case_id,
            runtime_id=runtime_case_id(case.case_id),
            true_root=case.ground_truth_root,
            evidence=evidence,
            board=board,
            graph=_trace_graph(catalog),
            symptomatic=symptomatic,
        ))
    return out


def _score(cases: list[LocalFusionCase], weights: FusionWeights) -> dict[str, float]:
    hit1 = mrr = 0.0
    for c in cases:
        result = fuse_candidate_evidence(c.evidence, c.board, weights=weights)
        result = topology_rerank(result, c.symptomatic or set(), c.graph, gamma=weights.topology_gamma)
        result = temporal_rerank(result, c.board, beta=weights.temporal_beta)
        ranked = [cand.component for cand in result.candidates]
        hit1 += 1.0 if accuracy_at_k(ranked, c.true_root, 1) else 0.0
        mrr += reciprocal_rank(ranked, c.true_root)
    n = max(1, len(cases))
    return {"hit_at_1": round(hit1 / n, 4), "mrr": round(mrr / n, 4), "n": len(cases)}


def _weights_for(boost: float, mod_bonus: float, gamma: float, beta: float, *,
                 version: str = "local_fusion_grid", fit_on: str = "", provenance: str = "") -> FusionWeights:
    return FusionWeights(
        version=version,
        fit_on=fit_on,
        provenance=provenance,
        modality_reliability_enabled=boost != 1.0,
        modality_weights={"metrics": 1.0, "logs": boost, "traces": boost,
                          "events": 1.0 + (boost - 1.0) / 2, "auxiliary": 0.8},
        convergence_modality_bonus=mod_bonus,
        topology_gamma=gamma,
        temporal_beta=beta,
    )


def fit(
    dev: list[LocalFusionCase],
    val: list[LocalFusionCase],
    *,
    boost_grid: tuple[float, ...] = (1.0,),
    convergence_grid: tuple[float, ...] = (0.18,),
    gamma_grid: tuple[float, ...] = (0.0, 2.0),
    beta_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
    fit_on: str = "rcaeval_hard_dev_offline",
    version: str = "local_fusion_fit_v3_temporal",
) -> tuple[FusionWeights, dict]:
    baseline = FusionWeights.default()
    dev_base = _score(dev, baseline)
    best = dev_base
    best_boost, best_bonus, best_gamma, best_beta = 1.0, baseline.convergence_modality_bonus, 0.0, 0.0
    trials = []
    for boost in boost_grid:
        for bonus in convergence_grid:
            for gamma in gamma_grid:
                for beta in beta_grid:
                    score = _score(dev, _weights_for(boost, bonus, gamma, beta))
                    trials.append({"boost": boost, "topology_gamma": gamma, "temporal_beta": beta, **score})
                    if (score["hit_at_1"], score["mrr"]) > (best["hit_at_1"], best["mrr"]):
                        best, best_boost, best_bonus, best_gamma, best_beta = score, boost, bonus, gamma, beta
    fitted = _weights_for(
        best_boost, best_bonus, best_gamma, best_beta,
        version=version, fit_on=fit_on,
        provenance=f"offline local-fusion re-rank grid on {len(dev)} dev cases",
    )
    graph_cases = sum(1 for c in dev if c.graph and c.graph.nodes)
    report = {
        "dev_baseline": dev_base,
        "dev_fitted": best,
        "val_baseline": _score(val, baseline),
        "val_fitted": _score(val, fitted),
        "selected": {"modality_boost": best_boost, "convergence_modality_bonus": best_bonus,
                     "topology_gamma": best_gamma, "temporal_beta": best_beta},
        "n_dev": len(dev),
        "n_val": len(val),
        "dev_cases_with_trace_graph": graph_cases,
        "trials": sorted(trials, key=lambda t: (-t["hit_at_1"], -t["mrr"]))[:12],
        "caveat": "offline local-fusion proxy for the live LLM synthesizer; prior only, not confirmatory",
    }
    return fitted, report


def _load_excluded(*prereg_paths: str) -> set[str]:
    excluded: set[str] = set()
    for path in prereg_paths:
        p = Path(path)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        ids = data.get("dataset", {}).get("runtime_case_ids", [])
        excluded.update(ids)
    return excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline local-fusion weight fit on a disjoint RCAEval-hard dev split")
    parser.add_argument("--n-dev", type=int, default=30)
    parser.add_argument("--n-val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--exclude", nargs="*", default=[
        "results/prereg_v7_holdout20_2026-07-03.json",
        "results/prereg_high_volume_draft.json",
    ], help="preregs whose runtime_case_ids must be held out of dev/val")
    parser.add_argument("--out-weights", default="results/weights/local_fusion_fit_v3_temporal.json")
    parser.add_argument("--out-report", default="results/local_fusion_fit_v3_report.json")
    parser.add_argument("--graph-only", action="store_true",
                        help="fit only on cases that have a trace dependency graph (fair test of topology)")
    args = parser.parse_args(argv)

    excluded = _load_excluded(*args.exclude)
    pool = [c for c in load_hard_cases(sample=None) if runtime_case_id(c.case_id) not in excluded]
    pool.sort(key=lambda c: c.case_id)
    import random
    random.Random(args.seed).shuffle(pool)
    need = args.n_dev + args.n_val
    if args.graph_only:
        # Cheap pre-filter (catalog listing only, no mining) to cases with traces.csv.
        pool = [c for c in pool if _has_trace_file(c)]
    chosen = pool[:need]
    extracted = extract_cases(chosen)
    print(json.dumps({"pool_after_exclusion": len(pool), "using": len(extracted),
                      "graph_only": args.graph_only, "excluded_ids": len(excluded)}))
    dev, val = extracted[: args.n_dev], extracted[args.n_dev : need]
    fitted, report = fit(dev, val)
    fitted.freeze(args.out_weights)
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps({"weights": fitted.to_dict(), "report": report}, indent=2))
    print(json.dumps({"out_weights": args.out_weights, "out_report": args.out_report, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

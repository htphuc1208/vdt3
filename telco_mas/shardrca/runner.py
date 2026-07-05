"""End-to-end ShardRCA runners for RCAEval and OpenRCA."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..evaluation.external import ExternalBenchmarkCase, ExternalPrediction
from ..llm import LLMClient
from ..openrca.schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput
from ..openrca.task_parser import OPENRCA_TIMEZONE, parse_runtime_task
from ..openrca.heuristic_floor import run_heuristic_floor
from ..openrca.tools import TELECOM_COMPONENTS, TELECOM_REASONS, candidate_catalog_for_row
from ..openrca.workers import refine_worker_distributions, run_isolated_workers, targeted_falsify
from ..schemas import UsageStats
from .board import Blackboard, CandidateRootCause
from .catalog import TelemetryCatalog, build_catalog, build_catalog_for_case, make_component_group_shards
from .falsifier import falsify
from .fusion import (
    candidate_evidence_from_findings,
    fuse_candidate_evidence,
    fuse_worker_distributions,
)
from .miner import MinerWorker
from .planner import plan_shards
from .weights import FusionWeights
from .single_baseline import run_single_baseline, run_single_react
from .synthesizer import SynthesizerResult, synthesize
from .temporal import temporal_rerank
from .topology import DependencyGraph, build_service_graph_from_trace_rows, topology_rerank

SHARDRCA_SYSTEMS = {
    "shardrca_full",
    "shardrca_local_fusion",
    "no_falsifier",
    "no_vote",
    "no_shard",
    "no_topology",
    "no_refinement",
}
SINGLE_SYSTEMS = {
    "single",
    "single_agent",
    "single_sc",
    "same_board_single",
    "single_react",
    "single_react_sc",
    "single_equal_tokens",
    "code_retrieval_single",
}
BUDGETED_SINGLE_SYSTEMS = {"single_react", "single_react_sc"}
EQUALIZED_SINGLE_SYSTEMS = {"single_equal_tokens"}
OPENRCA_AGENT_SYSTEMS = {"rca_agent_replica"}
OPENRCA_HEURISTIC_SYSTEMS = {"heuristic_floor"}


@dataclass
class ShardRCARunResult:
    system: str
    board: Blackboard
    synthesis: SynthesizerResult
    winner: CandidateRootCause
    usage: UsageStats
    latency_s: float
    notes: str = ""
    artifacts: dict[str, Any] | None = None


def run_catalog(
    catalog: TelemetryCatalog,
    *,
    system: str = "shardrca_full",
    task_payload: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
    chunksize: int = 50_000,
    finding_limit: int = 25,
) -> ShardRCARunResult:
    started = time.time()
    system = _normalise_system(system)
    if system in EQUALIZED_SINGLE_SYSTEMS:
        baseline = run_single_react(
            catalog,
            llm=llm,
            self_consistency=False,
            chunksize=chunksize,
            finding_limit=max(10, min(finding_limit, 20)),
            max_tool_calls=9,
        )
        return ShardRCARunResult(
            system=system,
            board=baseline.board,
            synthesis=baseline.synthesis,
            winner=baseline.winner,
            usage=baseline.usage,
            latency_s=round(time.time() - started, 2),
            notes="single-context ReAct baseline with expanded tool budget for compute sensitivity",
        )
    if system in BUDGETED_SINGLE_SYSTEMS:
        baseline = run_single_react(
            catalog,
            llm=llm,
            self_consistency=(system == "single_react_sc"),
            chunksize=chunksize,
            finding_limit=max(5, min(finding_limit, 10)),
            max_tool_calls=3,
        )
        return ShardRCARunResult(
            system=system,
            board=baseline.board,
            synthesis=baseline.synthesis,
            winner=baseline.winner,
            usage=baseline.usage,
            latency_s=round(time.time() - started, 2),
            notes="single ReAct baseline with budgeted local tools"
            + (" and self-consistency" if system == "single_react_sc" else ""),
        )
    if system in SINGLE_SYSTEMS:
        baseline = run_single_baseline(
            catalog,
            llm=llm,
            self_consistency=(system in {"single_sc", "same_board_single"}),
            chunksize=chunksize,
            finding_limit=finding_limit,
        )
        notes = (
            "single code-retrieval baseline over telemetry via deterministic miners"
            if system == "code_retrieval_single"
            else "single serial baseline" + (" with self-consistency" if system == "single_sc" else "")
        )
        return ShardRCARunResult(
            system=system,
            board=baseline.board,
            synthesis=baseline.synthesis,
            winner=baseline.winner,
            usage=baseline.usage,
            latency_s=round(time.time() - started, 2),
            notes=notes,
        )
    if system == "no_shard":
        baseline = run_single_baseline(
            catalog,
            llm=llm,
            self_consistency=True,
            chunksize=chunksize,
            finding_limit=finding_limit,
        )
        winner = baseline.winner
        usage = baseline.usage
        falsifier_result = falsify(baseline.board, baseline.synthesis.candidates, top=baseline.synthesis.winner)
        winner = falsifier_result.winner
        usage = usage.add(falsifier_result.usage)
        return ShardRCARunResult(
            system=system,
            board=baseline.board,
            synthesis=baseline.synthesis,
            winner=winner,
            usage=usage,
            latency_s=round(time.time() - started, 2),
            notes="serial no-shard ablation; " + falsifier_result.notes,
        )

    usage = UsageStats()
    shards = (
        make_component_group_shards(catalog, group_size=6)
        if str(catalog.dataset).startswith("RE")
        else plan_shards(catalog, task_payload=task_payload, llm=llm, max_shards=3)
    )
    board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
    worker = MinerWorker(limit=finding_limit, chunksize=chunksize)
    local_candidate_evidence = []
    for result in worker.run_many(shards):
        board.extend(result.findings)
        local_candidate_evidence.extend(candidate_evidence_from_findings(result.shard_id, result.findings))
        usage = usage.add(result.usage)

    if system == "shardrca_local_fusion":
        synthesis = fuse_candidate_evidence(local_candidate_evidence, board)
        return ShardRCARunResult(
            system=system,
            board=board,
            synthesis=synthesis,
            winner=synthesis.winner,
            usage=usage.add(synthesis.usage),
            latency_s=round(time.time() - started, 2),
            notes="fused local candidate evidence without the global-board LLM synthesizer",
        )

    synthesis = synthesize(board, llm=llm, k=1 if system == "no_vote" else 3)
    usage = usage.add(synthesis.usage)

    # Causal-structure re-rank (topology + temporal precedence). This is the
    # mechanism that lets an upstream root outrank a downstream symptom or a
    # backing datastore, which is exactly where the global-board single agent
    # otherwise beats the sharded fusion. Gated by frozen FusionWeights
    # (topology_gamma / temporal_beta); a default weights file is a no-op, and
    # the ``no_topology`` ablation disables it regardless.
    rerank_note = ""
    if system in {"shardrca_full", "no_falsifier", "no_topology", "no_refinement"}:
        weights = FusionWeights.load()
        rerank_enabled = system != "no_topology" and (
            weights.topology_gamma > 0.0 or weights.temporal_beta > 0.0
        )
        if rerank_enabled and synthesis.candidates:
            graph = _rcaeval_trace_graph(catalog)
            symptomatic = {
                component for component, score in board.top_components(20) if score > 0
            }
            synthesis = topology_rerank(
                synthesis, symptomatic, graph, gamma=weights.topology_gamma, max_candidates=8
            )
            synthesis = temporal_rerank(
                synthesis, board, beta=weights.temporal_beta, max_candidates=8
            )
            rerank_note = (
                f"; causal re-rank (gamma={weights.topology_gamma}, beta={weights.temporal_beta}, "
                f"graph_nodes={len(graph.nodes) if graph else 0})"
            )

    winner = synthesis.winner
    notes = "synthesized from sharded blackboard" + rerank_note
    if system != "no_falsifier":
        falsifier_result = falsify(board, synthesis.candidates, top=synthesis.winner)
        winner = falsifier_result.winner
        usage = usage.add(falsifier_result.usage)
        notes = falsifier_result.notes + rerank_note

    return ShardRCARunResult(
        system=system,
        board=board,
        synthesis=synthesis,
        winner=winner,
        usage=usage,
        latency_s=round(time.time() - started, 2),
        notes=notes,
    )


def run_rcaeval_case(
    case: ExternalBenchmarkCase,
    *,
    system: str = "shardrca_full",
    llm: LLMClient | None = None,
    chunksize: int = 50_000,
    finding_limit: int = 25,
) -> ExternalPrediction:
    catalog = build_catalog_for_case(case, compute_ranges=False)
    result = run_catalog(
        catalog,
        system=system,
        task_payload=case.inference_payload(),
        llm=llm,
        chunksize=chunksize,
        finding_limit=finding_limit,
    )
    ranked = _ranked_roots(result)
    winner = result.winner
    return ExternalPrediction(
        case_id=case.case_id,
        system=f"rcaeval_{_normalise_system(system)}",
        root=winner.component or (ranked[0] if ranked else "UNKNOWN"),
        ranked_roots=ranked or [winner.component],
        fault_type=winner.reason or "unknown",
        accepted=bool(winner.component and winner.component != "UNKNOWN"),
        confidence=winner.confidence,
        latency_s=result.latency_s,
        total_tokens=result.usage.total_tokens,
        tool_calls=result.usage.tool_calls,
        llm_calls=result.usage.llm_calls,
        notes=result.notes,
    )


def run_openrca_task(
    dataset,
    task: dict[str, Any],
    *,
    system: str = "shardrca_full",
    llm: LLMClient | None = None,
    chunksize: int = 50_000,
    finding_limit: int = 25,
    prepared=None,
) -> tuple[OpenRCAPredictionOutput, ShardRCARunResult]:
    parsed = parse_runtime_task(task)
    query_time_ms = (parsed.start_ms + parsed.end_ms) / 2.0
    if prepared is not None:
        catalog = prepared.build_catalog(parsed.row_id)
    else:
        date_root = dataset.telemetry_dir / parsed.date_key
        catalog = build_catalog(
            date_root,
            dataset=str(dataset.dataset),
            case_id=str(task.get("task_index") or task.get("row_id")),
            query_time=query_time_ms,
            time_range_start=parsed.start_ms,
            time_range_end=parsed.end_ms,
            compute_ranges=False,
        )
    normalized = _normalise_system(system)
    if normalized == "rca_agent_replica":
        if prepared is None or llm is None:
            raise ValueError("rca_agent_replica requires prepared telemetry and a live LLM")
        from ..openrca.rca_agent_replica import run_rca_agent_replica

        result = run_rca_agent_replica(prepared, task, llm=llm).run_result
    elif normalized == "heuristic_floor":
        if prepared is None:
            raise ValueError("heuristic_floor requires prepared telemetry")
        floor = run_heuristic_floor(
            prepared,
            parsed.row_id,
            candidate_catalog=candidate_catalog_for_row(prepared, parsed.row_id),
            finding_limit=max(10, finding_limit),
        )
        result = ShardRCARunResult(
            system=normalized,
            board=floor.board,
            synthesis=floor.synthesis,
            winner=floor.winner,
            usage=floor.usage,
            latency_s=0.0,
            notes=floor.notes,
            artifacts=floor.artifacts,
        )
    elif normalized in {"shardrca_full", "no_falsifier", "no_topology", "no_refinement"} and prepared is not None:
        started = time.time()
        candidate_catalog = candidate_catalog_for_row(prepared, parsed.row_id)
        workers = run_isolated_workers(
            prepared,
            parsed.row_id,
            llm=llm,
            max_workers=3,
            finding_limit=max(10, finding_limit),
            candidate_components=list(candidate_catalog["components"]),
            candidate_reasons=list(candidate_catalog["reasons"]),
            candidate_catalog_source=dict(candidate_catalog["source"]),
        )
        fusion_weights = FusionWeights.load()
        graph, topology_edges = _openrca_dependency_graph(prepared, parsed.row_id)
        symptomatic = {component for component, score in workers.board.top_components(20) if score > 0}
        round_1 = fuse_worker_distributions(
            workers.distributions,
            workers.board,
            components=list(candidate_catalog["components"]),
            reasons=list(candidate_catalog["reasons"]),
            weights=fusion_weights,
        )
        synthesis = _apply_openrca_reranks(
            round_1,
            workers.board,
            graph,
            symptomatic,
            weights=fusion_weights,
            enabled=normalized != "no_topology",
        )
        refinement_usage = UsageStats()
        refinement_diagnostic: dict[str, Any] = {
            "triggered": False,
            "reason": "disabled" if normalized == "no_refinement" else "not requested",
        }
        round_2 = None
        if normalized != "no_refinement":
            refined_distributions, refinement_usage, refinement_diagnostic = refine_worker_distributions(
                workers.distributions,
                workers.board,
                synthesis.candidates,
            )
            if refinement_diagnostic.get("triggered"):
                round_2 = fuse_worker_distributions(
                    refined_distributions,
                    workers.board,
                    components=list(candidate_catalog["components"]),
                    reasons=list(candidate_catalog["reasons"]),
                    weights=fusion_weights,
                )
                synthesis = _apply_openrca_reranks(
                    round_2,
                    workers.board,
                    graph,
                    symptomatic,
                    weights=fusion_weights,
                    enabled=normalized != "no_topology",
                )
        if normalized == "no_falsifier":
            winner = synthesis.winner
            falsifier_usage = UsageStats()
            falsifier_diagnostic = {"selected": "top", "reason": "disabled"}
        else:
            winner, falsifier_usage, falsifier_diagnostic = targeted_falsify(
                workers.board,
                synthesis.candidates,
                llm=llm,
            )
        result = ShardRCARunResult(
            system=normalized,
            board=workers.board,
            synthesis=synthesis,
            winner=winner,
            usage=workers.usage.add(synthesis.usage).add(refinement_usage).add(falsifier_usage),
            latency_s=round(time.time() - started, 2),
            notes=(
                "independent OpenRCA worker posteriors fused by correlation-aware "
                f"log-opinion pool (weights={fusion_weights.version})"
            ),
            artifacts={
                "candidate_catalog_source": candidate_catalog["source"],
                "candidate_catalog": {
                    "component_count": len(candidate_catalog["components"]),
                    "reason_count": len(candidate_catalog["reasons"]),
                    "components": list(candidate_catalog["components"]),
                    "reasons": list(candidate_catalog["reasons"]),
                },
                "board_findings": [finding.compact() for finding in workers.board.ranked_findings(160)],
                "worker_distributions": [distribution.compact() for distribution in workers.distributions],
                "worker_diagnostics": workers.diagnostics,
                "round_1": [candidate.compact() for candidate in round_1.candidates],
                "round_2": [candidate.compact() for candidate in round_2.candidates] if round_2 else [],
                "refinement": refinement_diagnostic,
                "topology": {
                    "enabled": normalized != "no_topology",
                    "edge_count": len(topology_edges),
                    "symptomatic_components": sorted(symptomatic)[:30],
                    "edges": topology_edges[:80],
                    "gamma": fusion_weights.topology_gamma,
                    "temporal_beta": fusion_weights.temporal_beta,
                },
                "fusion_candidates": [candidate.compact() for candidate in synthesis.candidates],
                "pre_falsifier_winner": synthesis.winner.compact(),
                "falsifier": falsifier_diagnostic,
                "fusion": "correlation_aware_log_opinion_pool",
                "fusion_weights": fusion_weights.to_dict(),
            },
        )
    else:
        result = run_catalog(
            catalog,
            system=system,
            task_payload={k: task[k] for k in ("row_id", "task_index", "instruction") if k in task},
            llm=llm,
            chunksize=chunksize,
            finding_limit=finding_limit,
        )
    requested = set(parsed.requested_fields)
    output_component = _openrca_output_component(result, requested)
    item = OpenRCAPredictionItem(
        root_cause_occurrence_datetime=(
            _format_openrca_occurrence(
                _openrca_occurrence_value(
                    result,
                    requested,
                    parsed.start_ms,
                    parsed.end_ms,
                    output_component,
                )
            )
            if "root cause occurrence datetime" in requested
            else None
        ),
        root_cause_component=(
            output_component
            if "root cause component" in requested and output_component != "UNKNOWN"
            else None
        ),
        root_cause_reason=(
            _openrca_reason(result.winner.reason)
            if "root cause reason" in requested
            else None
        ),
    )
    return OpenRCAPredictionOutput(root_causes=[item], rationale=result.notes or result.winner.rationale), result


def _normalise_system(system: str) -> str:
    if system in {"full", "multi", "multi_agent"}:
        return "shardrca_full"
    if system == "same_board_single":
        return "same_board_single"
    return system


def _apply_openrca_reranks(
    synthesis: SynthesizerResult,
    board: Blackboard,
    graph: DependencyGraph | None,
    symptomatic: set[str],
    *,
    weights: FusionWeights,
    enabled: bool,
) -> SynthesizerResult:
    if not enabled:
        return synthesis
    result = topology_rerank(
        synthesis,
        symptomatic,
        graph,
        gamma=weights.topology_gamma,
        max_candidates=8,
    )
    return temporal_rerank(
        result,
        board,
        beta=weights.temporal_beta,
        max_candidates=8,
    )


def _rcaeval_trace_graph(catalog: TelemetryCatalog, *, max_rows: int = 200_000) -> DependencyGraph | None:
    """Build a service dependency graph from an RCAEval case's raw traces.csv.

    Mirrors the offline fit path (fit_local_fusion._trace_graph): read only the
    span/parent/service columns and a bounded number of rows, since the call
    graph is a small low-cardinality structure recoverable from a span sample.
    Returns ``None`` when no trace file is present (e.g. RE2-SS cases), in which
    case topology re-rank is a no-op and only temporal precedence applies.
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


def _openrca_dependency_graph(prepared, row_id: int) -> tuple[DependencyGraph | None, list[tuple[str, str]]]:
    path = prepared.trace_edges_path(row_id)
    if path is None:
        return None, []
    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["parent_component", "child_component"])
    except Exception:
        return None, []
    edges = [
        (str(row["parent_component"]), str(row["child_component"]))
        for _, row in frame.iterrows()
        if str(row.get("parent_component") or "") and str(row.get("child_component") or "")
    ]
    return DependencyGraph.from_edges(edges), edges


def _openrca_occurrence_value(
    result: ShardRCARunResult,
    requested: set[str],
    query_start_ms: float,
    query_end_ms: float,
    output_component: str | None = None,
) -> str | float | int | None:
    """Choose a label-safe occurrence time for OpenRCA output.

    If component is not requested, tying the time to the predicted component can
    turn a good global onset estimate into a wrong answer. Prefer a conservative
    early-window causal prior for time-only/time+reason tasks; otherwise use the
    winner's own evidence timestamp.
    """

    if "root cause component" not in requested:
        prior_time = _query_window_early_prior_ms(query_start_ms, query_end_ms)
        if prior_time is not None:
            return prior_time
        board_time = _strongest_board_time(result.board)
        if board_time is not None:
            return board_time
    if output_component and output_component != result.winner.component:
        fallback_time = _strongest_component_time(result.board, output_component)
        if fallback_time is not None:
            return fallback_time
    if result.winner.occurrence_time is not None:
        return result.winner.occurrence_time
    winner_time = _strongest_component_time(result.board, result.winner.component)
    if winner_time is not None:
        return winner_time
    return _strongest_board_time(result.board) or _query_window_midpoint_ms(query_start_ms, query_end_ms)


def _query_window_early_prior_ms(start_ms: float, end_ms: float) -> float | None:
    if end_ms <= start_ms:
        return None
    return float(start_ms) + (float(end_ms) - float(start_ms)) / 3.0


def _query_window_midpoint_ms(start_ms: float, end_ms: float) -> float | None:
    if end_ms <= start_ms:
        return None
    return (float(start_ms) + float(end_ms)) / 2.0


def _openrca_output_component(result: ShardRCARunResult, requested: set[str]) -> str:
    component = result.winner.component
    if "root cause component" not in requested:
        return component
    if "root cause reason" in requested:
        return component
    if not _is_prior_only_winner(result):
        return component
    fallback = _strongest_catalog_component(result)
    return fallback or component


def _is_prior_only_winner(result: ShardRCARunResult) -> bool:
    if result.winner.evidence:
        return False
    rationale = (result.winner.rationale or "").lower()
    if "explicit_supporters=none" in rationale:
        return True
    return not bool(result.board.evidence_for(result.winner.component, limit=1))


def _strongest_catalog_component(result: ShardRCARunResult) -> str | None:
    allowed = set()
    artifacts = result.artifacts or {}
    catalog = artifacts.get("candidate_catalog")
    if isinstance(catalog, dict):
        allowed = {str(item) for item in catalog.get("components") or []}
    for finding in result.board.ranked_findings(80):
        component = finding.component
        if component == "UNKNOWN":
            continue
        if allowed and component not in allowed:
            continue
        return component
    for component, _score in result.board.top_components(20):
        if component == "UNKNOWN":
            continue
        if allowed and component not in allowed:
            continue
        return component
    return None


def _strongest_component_time(board: Blackboard, component: str) -> str | float | int | None:
    timed = [
        finding
        for finding in board.evidence_for(component, limit=20)
        if finding.window_start is not None
    ]
    if not timed:
        return None
    return max(timed, key=lambda finding: float(finding.score)).window_start


def _strongest_board_time(board: Blackboard) -> str | float | int | None:
    timed = [
        finding
        for finding in board.ranked_findings(80)
        if finding.window_start is not None
    ]
    if not timed:
        return None
    return max(timed, key=lambda finding: float(finding.score)).window_start


def _ranked_roots(result: ShardRCARunResult) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for candidate in sorted(result.synthesis.candidates, key=lambda item: (item.score, item.confidence), reverse=True):
        if candidate.component and candidate.component not in seen:
            ranked.append(candidate.component)
            seen.add(candidate.component)
    for component, _ in result.board.top_components(10):
        if component and component not in seen:
            ranked.append(component)
            seen.add(component)
    return ranked[:10]


def _extract_time_range(instruction: str) -> tuple[float | None, float | None]:
    matches = re.findall(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", instruction)
    if len(matches) >= 2:
        return _to_epoch(matches[0]), _to_epoch(matches[1])
    if len(matches) == 1:
        center = _to_epoch(matches[0])
        if center is None:
            return None, None
        return center - 30 * 60, center + 30 * 60
    return None, None


def _to_epoch(text: str) -> float | None:
    try:
        return float(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None


def _format_occurrence(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric > 1e15:
        numeric /= 1_000_000_000
    elif numeric > 1e12:
        numeric /= 1000
    try:
        return datetime.fromtimestamp(numeric, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _format_openrca_occurrence(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
        return value
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric > 1e15:
        numeric /= 1_000_000_000
    elif numeric > 1e12:
        numeric /= 1000
    try:
        return datetime.fromtimestamp(numeric, tz=OPENRCA_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _openrca_reason(reason: str | None) -> str | None:
    raw = (reason or "").strip()
    if raw in TELECOM_REASONS:
        return raw
    reason = raw.lower()
    if reason == "cpu":
        return "CPU fault"
    if reason in {"delay", "socket", "network"}:
        return "network delay"
    if reason == "loss":
        return "network loss"
    if reason == "db":
        return "db connection limit"
    return None

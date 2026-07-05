"""Evidence-isolated OpenRCA workers over prepared telemetry shards."""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from ..shardrca.board import (
    Blackboard,
    CandidateEvidence,
    CandidateRootCause,
    Finding,
    WorkerDistribution,
)
from .prepared import PreparedOpenRCA
from .tools import TELECOM_COMPONENTS, TELECOM_REASONS

WORKER_SCOPES: dict[str, dict[str, Any]] = {
    "node_metrics": {
        "modality": "metrics",
        "files": ("metric_node.parquet",),
        "components": tuple(component for component in TELECOM_COMPONENTS if component.startswith("os_")),
    },
    "container_metrics": {
        "modality": "metrics",
        "files": ("metric_container.parquet",),
        "components": tuple(component for component in TELECOM_COMPONENTS if component.startswith("docker_")),
    },
    "service_middleware_metrics": {
        "modality": "metrics",
        "files": ("metric_service.parquet", "metric_middleware.parquet"),
        "components": tuple(component for component in TELECOM_COMPONENTS if component.startswith("db_")),
    },
    "application_symptoms": {
        "modality": "metrics",
        "files": ("metric_app.parquet",),
        "components": tuple(TELECOM_COMPONENTS),
    },
    "trace_dependencies": {
        "modality": "traces",
        "files": (),
        "components": tuple(TELECOM_COMPONENTS),
    },
}

WORKER_SYSTEM = """You are one evidence-isolated OpenRCA investigator.
You may use ONLY the supplied local shard findings. You cannot see another worker or a global board.
Return a calibrated local distribution over the supplied official (component, reason) universe.
Every evidence pointer must be copied exactly from a supplied finding.
Return ONLY JSON:
{"candidates": [
  {"component": "official component", "reason": "official reason", "probability": 0.0,
   "support": 0.0, "refute": 0.0, "evidence_ptrs": ["..."],
   "missing_evidence": ["..."], "rationale": "short local rationale"}
 ], "other_mass": 0.0, "notes": "short"}
The candidate probabilities plus other_mass must sum to 1. Emit at most 8 candidates."""

FALSIFIER_SYSTEM = """You are the final OpenRCA falsifier.
You receive targeted prepared-telemetry evidence for exactly two candidates.
You may keep the top candidate or promote the runner-up. You may not invent a third candidate.
Return ONLY JSON:
{"selected": "top|runner_up", "top_status": "supported|refuted|inconclusive",
 "runner_up_status": "supported|refuted|inconclusive", "rationale": "short evidence-based reason"}"""


@dataclass
class WorkerRunResult:
    distributions: list[WorkerDistribution]
    board: Blackboard
    usage: UsageStats
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    candidate_catalog: dict[str, Any] = field(default_factory=dict)


def run_isolated_workers(
    prepared: PreparedOpenRCA,
    row_id: int,
    *,
    llm: LLMClient | None,
    max_workers: int = 3,
    finding_limit: int = 30,
    candidate_components: list[str] | None = None,
    candidate_reasons: list[str] | None = None,
    candidate_catalog_source: dict[str, Any] | None = None,
) -> WorkerRunResult:
    components = list(candidate_components or TELECOM_COMPONENTS)
    reasons = list(candidate_reasons or TELECOM_REASONS)
    scopes = _worker_scopes(components)
    board = Blackboard(
        case_id=str(row_id),
        catalog_summary={
            "dataset": "OpenRCA Telecom",
            "prepared": True,
            "row_id": row_id,
            "volume": prepared.volume(row_id),
            "candidate_catalog_source": candidate_catalog_source or {
                "components": "legacy_protocol_prior",
                "reasons": "protocol_prior_openrca_reason_catalog",
                "label_derived": False,
            },
        },
    )
    payloads: dict[str, dict[str, Any]] = {}
    for worker_id, spec in scopes.items():
        findings = _findings_for_worker(prepared, row_id, worker_id, limit=finding_limit)
        board.extend(findings)
        observed_components = sorted({
            finding.component
            for finding in findings
            if finding.component in set(spec["components"])
        })
        payloads[worker_id] = {
            "worker_id": worker_id,
            "modality": spec["modality"],
            "candidate_components": observed_components,
            "candidate_reasons": reasons,
            "findings": [finding.compact() for finding in findings],
            "constraint": "Use only this shard. Candidate labels are a fixed closed-set catalog.",
        }

    usage = UsageStats()
    diagnostics: list[dict[str, Any]] = []
    distributions: dict[str, WorkerDistribution] = {}
    if llm is None:
        for worker_id, payload in payloads.items():
            distributions[worker_id] = _fallback_distribution(worker_id, payload, board)
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(payloads)))) as executor:
            futures = {
                executor.submit(_run_one_worker, llm, worker_id, payload, board): worker_id
                for worker_id, payload in payloads.items()
            }
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    distribution, worker_usage, diagnostic = future.result()
                except Exception as exc:
                    distribution = _fallback_distribution(worker_id, payloads[worker_id], board)
                    worker_usage = UsageStats()
                    diagnostic = {"worker_id": worker_id, "fallback": True, "error": str(exc)}
                distributions[worker_id] = distribution
                usage = usage.add(worker_usage)
                diagnostics.append(diagnostic)
    return WorkerRunResult(
        distributions=[distributions[worker_id] for worker_id in scopes],
        board=board,
        usage=usage,
        diagnostics=sorted(diagnostics, key=lambda item: str(item.get("worker_id"))),
        candidate_catalog={
            "components": components,
            "reasons": reasons,
            "source": candidate_catalog_source or {
                "components": "legacy_protocol_prior",
                "reasons": "protocol_prior_openrca_reason_catalog",
                "label_derived": False,
            },
        },
    )


def targeted_falsify(
    board: Blackboard,
    candidates: list[CandidateRootCause],
    *,
    llm: LLMClient | None,
) -> tuple[CandidateRootCause, UsageStats, dict[str, Any]]:
    if not candidates:
        unknown = CandidateRootCause(component="UNKNOWN", reason="unknown")
        return unknown, UsageStats(), {"selected": "none", "reason": "no candidates"}
    top = candidates[0]
    runner = next((candidate for candidate in candidates[1:] if candidate.component != top.component), None)
    top_check = _candidate_targeted_check(board, top)
    runner_check = _candidate_targeted_check(board, runner) if runner is not None else None
    if runner is None:
        return top, UsageStats(tool_calls=1), {
            "selected": "top",
            "reason": "no runner-up candidate",
            "top": top_check,
        }
    deterministic = _select_by_targeted_checks(top, runner, top_check, runner_check)
    if llm is None:
        return deterministic, UsageStats(tool_calls=2), {
            "selected": "runner_up" if deterministic is runner else "top",
            "reason": "deterministic targeted evidence check",
            "top": top_check,
            "runner_up": runner_check,
        }
    payload = {
        "top": top.compact(),
        "runner_up": runner.compact(),
        "targeted_evidence": {
            "top": top_check,
            "runner_up": runner_check,
        },
        "deterministic_recommendation": "runner_up" if deterministic is runner else "top",
        "allowed_selection": ["top", "runner_up"],
    }
    response = llm.chat(
        [
            {"role": "system", "content": FALSIFIER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        force_json=True,
    )
    data = extract_json(response.content)
    selected = str(data.get("selected") or "top")
    winner = runner if selected == "runner_up" else top
    diagnostic = {
        "selected": "runner_up" if winner is runner else "top",
        "top_status": str(data.get("top_status") or top_check["status"]).upper(),
        "runner_up_status": str(data.get("runner_up_status") or runner_check["status"]).upper(),
        "top": top_check,
        "runner_up": runner_check,
        "rationale": str(data.get("rationale") or ""),
    }
    return winner, response.usage, diagnostic


def refine_worker_distributions(
    distributions: list[WorkerDistribution],
    board: Blackboard,
    candidates: list[CandidateRootCause],
    *,
    max_requests: int = 2,
    margin_threshold: float = 0.05,
) -> tuple[list[WorkerDistribution], UsageStats, dict[str, Any]]:
    """Add a bounded top-vs-runner refinement distribution when fusion is unstable."""

    if len(candidates) < 2:
        return distributions, UsageStats(), {
            "triggered": False,
            "reason": "fewer than two candidates",
            "requests": [],
        }
    top = candidates[0]
    runner = next((candidate for candidate in candidates[1:] if candidate.component != top.component), None)
    if runner is None:
        return distributions, UsageStats(), {
            "triggered": False,
            "reason": "no distinct runner-up",
            "requests": [],
        }
    margin = float(top.score) - float(runner.score)
    conflict = _worker_conflict(distributions)
    if margin > margin_threshold and not conflict["conflict"]:
        return distributions, UsageStats(), {
            "triggered": False,
            "reason": "winner margin is stable and workers do not conflict",
            "margin": round(margin, 6),
            "conflict": conflict,
            "requests": [],
        }

    top_check = _candidate_targeted_check(board, top)
    runner_check = _candidate_targeted_check(board, runner)
    requested_workers = _refinement_workers(distributions, {top.component, runner.component})[:max_requests]
    refinement = _refinement_distribution(top, runner, top_check, runner_check)
    diagnostics = {
        "triggered": True,
        "reason": "low margin or cross-worker conflict",
        "margin": round(margin, 6),
        "conflict": conflict,
        "requests": [
            {
                "worker_id": worker_id,
                "question": f"Re-check evidence separating {top.component} from {runner.component}.",
            }
            for worker_id in requested_workers
        ],
        "checks": {"top": top_check, "runner_up": runner_check},
    }
    return distributions + [refinement], UsageStats(tool_calls=max(1, len(requested_workers))), diagnostics


def _worker_scopes(components: list[str]) -> dict[str, dict[str, Any]]:
    component_set = sorted(dict.fromkeys(str(component) for component in components if str(component).strip()))
    if not component_set:
        component_set = list(TELECOM_COMPONENTS)
    return {
        "node_metrics": {
            "modality": "metrics",
            "files": ("metric_node.parquet",),
            "components": tuple(component for component in component_set if component.startswith("os_")),
        },
        "container_metrics": {
            "modality": "metrics",
            "files": ("metric_container.parquet",),
            "components": tuple(component for component in component_set if component.startswith("docker_")),
        },
        "service_middleware_metrics": {
            "modality": "metrics",
            "files": ("metric_service.parquet", "metric_middleware.parquet"),
            "components": tuple(component for component in component_set if component.startswith("db_")),
        },
        "application_symptoms": {
            "modality": "metrics",
            "files": ("metric_app.parquet",),
            "components": tuple(component_set),
        },
        "trace_dependencies": {
            "modality": "traces",
            "files": (),
            "components": tuple(component_set),
        },
    }


def _candidate_targeted_check(board: Blackboard, candidate: CandidateRootCause) -> dict[str, Any]:
    evidence = board.evidence_for(candidate.component, limit=12)
    reason = str(candidate.reason or "").strip()
    support = 0.0
    refute = 0.0
    ptrs: list[str] = []
    counter_ptrs: list[str] = []
    for finding in evidence:
        score = max(0.0, float(finding.score))
        hint = str(finding.metadata.get("reason_hint") or "").strip()
        if finding.evidence_ptr:
            ptrs.append(finding.evidence_ptr)
        if not reason or reason == "unknown" or not hint or hint == reason:
            support += score
        else:
            support += 0.25 * score
            refute += 0.5 * score
            if finding.evidence_ptr:
                counter_ptrs.append(finding.evidence_ptr)
    if not evidence:
        status = "REFUTED"
    elif support >= max(0.05, refute * 1.25):
        status = "SUPPORTED"
    elif refute > support:
        status = "REFUTED"
    else:
        status = "INCONCLUSIVE"
    return {
        "candidate": candidate.compact(),
        "status": status,
        "support_score": round(support, 6),
        "refute_score": round(refute, 6),
        "evidence_ptrs": ptrs[:8],
        "counter_evidence_ptrs": counter_ptrs[:8],
    }


def _select_by_targeted_checks(
    top: CandidateRootCause,
    runner: CandidateRootCause,
    top_check: dict[str, Any],
    runner_check: dict[str, Any],
) -> CandidateRootCause:
    top_support = float(top_check.get("support_score") or 0.0)
    runner_support = float(runner_check.get("support_score") or 0.0)
    top_status = str(top_check.get("status") or "")
    runner_status = str(runner_check.get("status") or "")
    if runner_status == "SUPPORTED" and top_status == "REFUTED":
        return runner
    if runner_status == "SUPPORTED" and runner_support > top_support + 0.05:
        return runner
    return top


def _worker_conflict(distributions: list[WorkerDistribution]) -> dict[str, Any]:
    top_components = []
    for distribution in distributions:
        if distribution.candidates:
            top_components.append(distribution.candidates[0].component)
    unique = sorted({component for component in top_components if component})
    return {
        "conflict": len(unique) > 1,
        "top_components": top_components,
        "unique_top_components": unique,
    }


def _refinement_workers(distributions: list[WorkerDistribution], components: set[str]) -> list[str]:
    out = []
    needles = {component.strip().lower() for component in components}
    for distribution in distributions:
        if any(candidate.component.strip().lower() in needles for candidate in distribution.candidates):
            out.append(distribution.worker_id)
    return out or [distribution.worker_id for distribution in distributions[:2]]


def _refinement_distribution(
    top: CandidateRootCause,
    runner: CandidateRootCause,
    top_check: dict[str, Any],
    runner_check: dict[str, Any],
) -> WorkerDistribution:
    raw = [
        (top, max(0.0, float(top_check.get("support_score") or 0.0) - float(top_check.get("refute_score") or 0.0)), top_check),
        (
            runner,
            max(0.0, float(runner_check.get("support_score") or 0.0) - float(runner_check.get("refute_score") or 0.0)),
            runner_check,
        ),
    ]
    total = sum(score for _, score, _ in raw)
    explicit_mass = 0.9 if total > 0 else 0.0
    candidates = []
    for rank, (candidate, score, check) in enumerate(raw, start=1):
        probability = explicit_mass * score / total if total > 0 else 0.0
        candidates.append(
            CandidateEvidence(
                component=candidate.component,
                reason_family=candidate.reason,
                probability=probability,
                support_score=float(check.get("support_score") or 0.0),
                refute_score=float(check.get("refute_score") or 0.0),
                modality="auxiliary",
                worker_id="iterative_refinement",
                shard_id="iterative_refinement",
                evidence_ptrs=list(check.get("evidence_ptrs") or [])[:6],
                local_rank=rank,
                rationale=f"top-vs-runner targeted check status={check.get('status')}",
            )
        )
    return WorkerDistribution(
        worker_id="iterative_refinement",
        modality="auxiliary",
        candidate_scope=[top.component, runner.component],
        candidates=candidates,
        other_mass=1.0 - explicit_mass,
        notes="bounded top-vs-runner refinement distribution",
    )


def _run_one_worker(
    llm: LLMClient,
    worker_id: str,
    payload: dict[str, Any],
    board: Blackboard,
) -> tuple[WorkerDistribution, UsageStats, dict[str, Any]]:
    response = llm.chat(
        [
            {"role": "system", "content": WORKER_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        force_json=True,
    )
    data = extract_json(response.content)
    distribution, errors = _parse_distribution(worker_id, payload, data)
    if distribution is None:
        distribution = _fallback_distribution(worker_id, payload, board)
    return distribution, response.usage, {
        "worker_id": worker_id,
        "fallback": bool(errors),
        "validation_errors": errors,
    }


def _parse_distribution(
    worker_id: str,
    payload: dict[str, Any],
    data: dict[str, Any],
) -> tuple[WorkerDistribution | None, list[str]]:
    errors: list[str] = []
    scope = list(payload["candidate_components"])
    allowed_components = set(scope)
    allowed_reasons = set(payload.get("candidate_reasons") or TELECOM_REASONS)
    allowed_ptrs = {
        str(finding.get("evidence") or "")
        for finding in payload.get("findings", [])
        if finding.get("evidence")
    }
    candidates: list[CandidateEvidence] = []
    seen: set[tuple[str, str]] = set()
    for raw in (data.get("candidates") or [])[:8]:
        if not isinstance(raw, dict):
            errors.append("non-object candidate")
            continue
        component = str(raw.get("component") or "")
        reason = str(raw.get("reason") or "")
        pair = (component, reason)
        if component not in allowed_components or reason not in allowed_reasons or pair in seen:
            errors.append(f"invalid candidate {component}|{reason}")
            continue
        try:
            probability = float(raw.get("probability") or 0.0)
            support = float(raw.get("support") or 0.0)
            refute = float(raw.get("refute") or 0.0)
        except Exception:
            errors.append(f"non-numeric candidate {component}|{reason}")
            continue
        pointers = [str(item) for item in raw.get("evidence_ptrs", []) if str(item) in allowed_ptrs]
        if len(pointers) != len(raw.get("evidence_ptrs", []) or []):
            errors.append(f"invalid evidence pointer for {component}|{reason}")
        candidates.append(
            CandidateEvidence(
                component=component,
                reason_family=reason,
                probability=max(0.0, min(1.0, probability)),
                support_score=support,
                refute_score=refute,
                modality=payload["modality"],
                worker_id=worker_id,
                shard_id=worker_id,
                evidence_ptrs=pointers,
                missing_evidence=[str(item) for item in raw.get("missing_evidence", [])[:6]],
                local_rank=len(candidates) + 1,
                rationale=str(raw.get("rationale") or ""),
            )
        )
        seen.add(pair)
    try:
        other_mass = max(0.0, min(1.0, float(data.get("other_mass") or 0.0)))
    except Exception:
        other_mass = 0.0
        errors.append("invalid other_mass")
    total = sum(item.probability for item in candidates) + other_mass
    if not candidates and other_mass <= 0:
        errors.append("empty distribution")
        return None, errors
    if total <= 0:
        return None, errors + ["zero distribution mass"]
    for item in candidates:
        item.probability /= total
    other_mass /= total
    return WorkerDistribution(
        worker_id=worker_id,
        modality=payload["modality"],
        candidate_scope=scope,
        candidates=candidates,
        other_mass=other_mass,
        notes=str(data.get("notes") or ""),
    ), errors


def _fallback_distribution(
    worker_id: str,
    payload: dict[str, Any],
    board: Blackboard,
) -> WorkerDistribution:
    scope = set(payload["candidate_components"])
    findings = [
        finding
        for finding in board.findings
        if finding.shard_id == worker_id and finding.component in scope
    ]
    scores: dict[tuple[str, str], float] = {}
    evidence: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        reason = str(finding.metadata.get("reason_hint") or "")
        if reason not in set(payload.get("candidate_reasons") or TELECOM_REASONS):
            continue
        pair = (finding.component, reason)
        scores[pair] = scores.get(pair, 0.0) + max(0.0, float(finding.score))
        if finding.evidence_ptr:
            evidence.setdefault(pair, []).append(finding.evidence_ptr)
    if not scores:
        return WorkerDistribution(
            worker_id=worker_id,
            modality=payload["modality"],
            candidate_scope=list(payload["candidate_components"]),
            candidates=[],
            other_mass=1.0,
            notes="deterministic fallback with no scoped candidate evidence",
        )
    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]
    # Board scores are open-scale anomaly magnitudes (e.g. 72, 70, 11). The old
    # ``exp(min(20.0, score))`` saturated every strong anomaly to the same cap,
    # collapsing the distribution to near-uniform and destroying the ranking
    # before fusion/topology re-rank could use it. Normalise by the max score so
    # the softmax is scale-invariant (works for both z-score-scale RCAEval and
    # large-magnitude telecom scores) and preserves the relative anomaly order.
    max_score = max((score for _, score in top), default=0.0)
    tau = 0.3
    if max_score > 0.0:
        exponentials = [math.exp((score / max_score) / tau) for _, score in top]
    else:
        exponentials = [1.0 for _ in top]
    normalizer = sum(exponentials) or 1.0
    explicit_mass = 0.9
    candidates = []
    for rank, (((component, reason), score), value) in enumerate(zip(top, exponentials), start=1):
        candidates.append(
            CandidateEvidence(
                component=component,
                reason_family=reason,
                probability=explicit_mass * value / normalizer,
                support_score=score,
                modality=payload["modality"],
                worker_id=worker_id,
                shard_id=worker_id,
                evidence_ptrs=evidence.get((component, reason), [])[:6],
                local_rank=rank,
                rationale="deterministic prepared-telemetry fallback",
            )
        )
    return WorkerDistribution(
        worker_id=worker_id,
        modality=payload["modality"],
        candidate_scope=list(payload["candidate_components"]),
        candidates=candidates,
        other_mass=1.0 - explicit_mass,
        notes="deterministic fallback distribution",
    )


def _findings_for_worker(
    prepared: PreparedOpenRCA,
    row_id: int,
    worker_id: str,
    *,
    limit: int,
) -> list[Finding]:
    if worker_id == "trace_dependencies":
        return _trace_findings(prepared, row_id, worker_id, limit=limit)
    return _metric_findings(prepared, row_id, worker_id, limit=limit)


def _metric_findings(
    prepared: PreparedOpenRCA,
    row_id: int,
    worker_id: str,
    *,
    limit: int,
) -> list[Finding]:
    import pandas as pd

    spec = WORKER_SCOPES[worker_id]
    task = prepared.task(row_id)
    stats_path = prepared.metric_stats_path(row_id)
    stats = pd.read_parquet(stats_path) if stats_path.exists() else pd.DataFrame()
    findings: list[Finding] = []
    for filename in spec["files"]:
        path = prepared.row_dir(row_id) / "metric" / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        source = filename.replace(".parquet", ".csv")
        if filename == "metric_app.parquet":
            findings.extend(_wide_metric_findings(frame, stats, worker_id, path, task))
            continue
        required = {"cmdb_id", "name", "value"}
        if not required.issubset(frame.columns):
            continue
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        time_col = "timestamp" if "timestamp" in frame else "startTime"
        frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
        local_stats = stats[stats.get("source_file", "") == source] if not stats.empty else pd.DataFrame()
        for (component, metric), group in frame.groupby(["cmdb_id", "name"], dropna=True):
            component = str(component)
            metric = str(metric)
            baseline = local_stats[
                (local_stats["component"].astype(str) == component)
                & (local_stats["metric"].astype(str) == metric)
            ]
            if baseline.empty:
                continue
            row = baseline.iloc[0]
            finding = _anomaly_finding(
                group,
                component=component,
                metric=metric,
                value_col="value",
                time_col=time_col,
                p05=float(row.get("p05") or 0.0),
                p95=float(row.get("p95") or 0.0),
                worker_id=worker_id,
                modality="metrics",
                evidence_ptr=f"rows/{row_id:03d}/metric/{filename}#{component}|{metric}",
            )
            if finding is not None:
                findings.append(finding)
    return sorted(findings, key=lambda item: (item.score, item.magnitude), reverse=True)[:limit]


def _wide_metric_findings(frame, stats, worker_id: str, path: Path, task: dict[str, Any]) -> list[Finding]:
    import pandas as pd

    if "serviceName" not in frame:
        return []
    time_col = "startTime" if "startTime" in frame else "timestamp"
    if time_col not in frame:
        return []
    frame[time_col] = pd.to_numeric(frame[time_col], errors="coerce")
    findings = []
    for metric in ("avg_time", "num", "succee_num", "succee_rate"):
        if metric not in frame:
            continue
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        for component, group in frame.groupby("serviceName", dropna=True):
            baseline = stats[
                (stats["source_file"] == "metric_app.csv")
                & (stats["component"].astype(str) == str(component))
                & (stats["metric"].astype(str) == metric)
            ]
            if baseline.empty:
                continue
            row = baseline.iloc[0]
            finding = _anomaly_finding(
                group,
                component=str(component),
                metric=metric,
                value_col=metric,
                time_col=time_col,
                p05=float(row.get("p05") or 0.0),
                p95=float(row.get("p95") or 0.0),
                worker_id=worker_id,
                modality="metrics",
                evidence_ptr=f"rows/{int(task['row_id']):03d}/metric/{path.name}#{component}|{metric}",
            )
            if finding is not None:
                findings.append(finding)
    return findings


def _anomaly_finding(
    group,
    *,
    component: str,
    metric: str,
    value_col: str,
    time_col: str,
    p05: float,
    p95: float,
    worker_id: str,
    modality: str,
    evidence_ptr: str,
) -> Finding | None:
    values = group[value_col].dropna()
    if values.empty:
        return None
    high = max(0.0, float(values.max()) - p95) / (abs(p95) + 1.0)
    low = max(0.0, p05 - float(values.min())) / (abs(p05) + 1.0)
    if high <= 0 and low <= 0:
        return None
    direction = "high" if high >= low else "low"
    score = max(high, low)
    anomalous = group[group[value_col] > p95] if direction == "high" else group[group[value_col] < p05]
    occurrence = None
    if not anomalous.empty:
        if direction == "high":
            index = anomalous[value_col].astype(float).idxmax()
        else:
            index = anomalous[value_col].astype(float).idxmin()
        occurrence = float(anomalous.loc[index, time_col])
    hint = _reason_hint(metric, direction)
    return Finding(
        shard_id=worker_id,
        modality=modality,  # type: ignore[arg-type]
        component=component,
        signal=metric,
        direction=direction,
        magnitude=score,
        score=score,
        window_start=occurrence,
        evidence_ptr=evidence_ptr,
        summary=f"{metric} crossed daily p05/p95; direction={direction}; severity={score:.4f}",
        metadata={"p05": p05, "p95": p95, "reason_hint": hint},
    )


def _trace_findings(
    prepared: PreparedOpenRCA,
    row_id: int,
    worker_id: str,
    *,
    limit: int,
) -> list[Finding]:
    import pandas as pd

    summary_path = prepared.trace_summary_path(row_id)
    if summary_path is None:
        return []
    frame = pd.read_parquet(summary_path)
    if frame.empty:
        return []
    global_median = float(pd.to_numeric(frame["mean"], errors="coerce").median() or 0.0)
    findings = []
    for index, row in frame.iterrows():
        component = _first_text(row.get("cmdb_id"), row.get("serviceName"))
        if not component:
            continue
        mean = float(row.get("mean") or 0.0)
        maximum = float(row.get("max") or 0.0)
        score = max(0.0, mean - global_median) / (abs(global_median) + 1.0)
        score += max(0.0, maximum - mean) / (abs(mean) + 1.0) * 0.1
        if score <= 0:
            continue
        signal = _first_text(row.get("dsName"), row.get("callType")) or "trace_latency"
        failure_count = int(row.get("failure_count") or 0)
        reason_hint = "network loss" if failure_count > 0 else "network delay"
        occurrence = row.get("max_elapsed_start") or row.get("first_start")
        findings.append(
            Finding(
                shard_id=worker_id,
                modality="traces",
                component=component,
                signal=signal,
                direction="high",
                magnitude=score,
                score=score,
                window_start=occurrence,
                evidence_ptr=f"rows/{row_id:03d}/trace_summary.parquet#row={index}",
                summary=f"trace latency mean={mean:.4f} max={maximum:.4f} count={int(row.get('count') or 0)}",
                metadata={"reason_hint": reason_hint, "failure_count": failure_count},
            )
        )
    return sorted(findings, key=lambda item: item.score, reverse=True)[:limit]


def _reason_hint(metric: str, direction: str) -> str:
    text = metric.lower()
    if any(token in text for token in ("success", "succee", "loss", "drop", "error", "fail")):
        return "network loss"
    if any(token in text for token in ("close", "closed", "on_off_state", "onoff")):
        return "db close"
    if any(token in text for token in ("connect", "client", "session", "login", "blocked", "rejected", "row_lock", "hang")):
        return "db connection limit"
    if any(token in text for token in ("cpu", "processor", "load")):
        return "CPU fault"
    if any(token in text for token in ("delay", "latency", "elapsed", "time")):
        return "network delay"
    return ""


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text.lower() not in {"nan", "<na>", "none"}:
            return text
    return ""

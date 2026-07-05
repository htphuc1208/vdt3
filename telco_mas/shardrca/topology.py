"""Topology-aware re-ranking for ShardRCA fusion.

Motivation: fusion over per-component evidence has no notion of causal structure,
so it cannot tell an upstream cause from a downstream symptom. In microservice /
telecom dependency graphs a fault propagates from the faulty node to the things
that depend on it (callee -> caller), so the true root tends to be the node whose
failure *explains the most symptomatic nodes* among itself and its (transitive)
callers. This module scores that explanatory coverage on a dependency graph and
re-ranks fused candidates by it.

The re-rank is gated by ``FusionWeights.topology_gamma`` (default 0.0 = no-op), so
it changes nothing until a validation fit turns it on. Unlike ``correlation_rho``
and ``temperature`` (monotone, argmax-preserving), this can change the winner.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .board import CandidateRootCause
from .synthesizer import SynthesizerResult


@dataclass
class DependencyGraph:
    """Directed call graph: an edge caller -> callee means caller depends on callee."""

    calls: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    called_by: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    nodes: set[str] = field(default_factory=set)

    @classmethod
    def from_edges(cls, edges) -> "DependencyGraph":
        graph = cls(calls=defaultdict(set), called_by=defaultdict(set), nodes=set())
        for caller, callee in edges:
            caller = str(caller).strip().lower()
            callee = str(callee).strip().lower()
            if not caller or not callee or caller == callee:
                continue
            graph.calls[caller].add(callee)
            graph.called_by[callee].add(caller)
            graph.nodes.add(caller)
            graph.nodes.add(callee)
        return graph

    def ancestors(self, node: str) -> set[str]:
        """Transitive callers of ``node`` (who a failure at ``node`` would affect)."""
        node = node.strip().lower()
        seen: set[str] = set()
        stack = list(self.called_by.get(node, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.called_by.get(current, set()))
        return seen


def explanatory_coverage(node: str, symptomatic: set[str], graph: DependencyGraph) -> float:
    """Fraction of symptomatic nodes explained by a failure at ``node``.

    Explained = the node itself plus its transitive callers (propagation targets).
    Returns 0 when the node is not in the graph or there are no symptoms.
    """
    node = node.strip().lower()
    symptomatic = {s.strip().lower() for s in symptomatic}
    if not symptomatic or node not in graph.nodes:
        return 0.0
    closure = {node} | graph.ancestors(node)
    return len(symptomatic & closure) / len(symptomatic)


def topology_rerank(
    result: SynthesizerResult,
    symptomatic: set[str],
    graph: DependencyGraph | None,
    *,
    gamma: float,
    max_candidates: int = 8,
) -> SynthesizerResult:
    """Re-rank fused candidates by ``score * (1 + gamma * explanatory_coverage)``.

    ``gamma == 0`` or a missing/empty graph returns the input unchanged.
    """
    if gamma <= 0.0 or graph is None or not graph.nodes or not result.candidates:
        return result

    rescored: list[tuple[CandidateRootCause, float]] = []
    for cand in result.candidates:
        coverage = explanatory_coverage(cand.component, symptomatic, graph)
        new_score = float(cand.score) * (1.0 + gamma * coverage)
        rescored.append((cand, new_score))
    rescored.sort(key=lambda item: item[1], reverse=True)

    total = sum(score for _, score in rescored) or 1.0
    candidates: list[CandidateRootCause] = []
    for cand, score in rescored[:max_candidates]:
        candidates.append(cand.model_copy(update={
            "score": round(score, 6),
            "confidence": max(0.05, min(0.95, score / total)),
        }))
    return SynthesizerResult(
        winner=candidates[0],
        candidates=candidates,
        vote_breakdown={c.component: round(c.score, 4) for c in candidates},
        usage=result.usage,
    )


def build_service_graph_from_trace_rows(rows) -> DependencyGraph:
    """Build a call graph from RCAEval-style trace rows.

    ``rows`` is an iterable of dicts/records with ``spanID``, ``parentSpanID`` and
    ``serviceName``. A parent span's service calls the child span's service.
    """
    span_service: dict[str, str] = {}
    parent_of: list[tuple[str, str]] = []
    for row in rows:
        span_id = str(row.get("spanID") or row.get("spanId") or "")
        parent_id = str(row.get("parentSpanID") or row.get("parentSpanId") or "")
        service = str(row.get("serviceName") or row.get("service") or "")
        if span_id and service:
            span_service[span_id] = service
        if parent_id and span_id:
            parent_of.append((parent_id, span_id))
    edges = []
    for parent_span, child_span in parent_of:
        caller = span_service.get(parent_span)
        callee = span_service.get(child_span)
        if caller and callee:
            edges.append((caller, callee))
    return DependencyGraph.from_edges(edges)

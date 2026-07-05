"""Topology-aware re-ranking: an upstream cause should outrank a downstream symptom."""
from __future__ import annotations

from telco_mas.shardrca.board import CandidateRootCause
from telco_mas.shardrca.synthesizer import SynthesizerResult
from telco_mas.shardrca.topology import (
    DependencyGraph,
    build_service_graph_from_trace_rows,
    explanatory_coverage,
    topology_rerank,
)

# frontend -> checkout -> emailservice (emailservice is the leaf callee / true root)
EDGES = [("frontend", "checkout"), ("checkout", "emailservice"), ("frontend", "emailservice")]
GRAPH = DependencyGraph.from_edges(EDGES)
SYMPTOMATIC = {"emailservice", "checkout", "frontend"}


def test_ancestors_are_transitive_callers():
    assert GRAPH.ancestors("emailservice") == {"checkout", "frontend"}
    assert GRAPH.ancestors("frontend") == set()


def test_leaf_root_has_highest_coverage():
    cov_root = explanatory_coverage("emailservice", SYMPTOMATIC, GRAPH)
    cov_mid = explanatory_coverage("checkout", SYMPTOMATIC, GRAPH)
    cov_top = explanatory_coverage("frontend", SYMPTOMATIC, GRAPH)
    assert cov_root == 1.0            # itself + both callers explain all symptoms
    assert cov_root > cov_mid > cov_top


def _result():
    # Fusion (evidence only) wrongly ranks the symptomatic caller 'frontend' first.
    return SynthesizerResult(
        winner=CandidateRootCause(component="frontend", score=1.0),
        candidates=[
            CandidateRootCause(component="frontend", score=1.0),
            CandidateRootCause(component="emailservice", score=0.85),
        ],
    )


def test_gamma_zero_is_noop():
    out = topology_rerank(_result(), SYMPTOMATIC, GRAPH, gamma=0.0)
    assert out.winner.component == "frontend"


def test_topology_flips_symptom_to_true_root():
    out = topology_rerank(_result(), SYMPTOMATIC, GRAPH, gamma=2.0)
    # emailservice: 0.85*(1+2*1.0)=2.55 ; frontend: 1.0*(1+2*(1/3))=1.667
    assert out.winner.component == "emailservice"


def test_missing_graph_is_noop():
    out = topology_rerank(_result(), SYMPTOMATIC, None, gamma=2.0)
    assert out.winner.component == "frontend"


def test_graph_from_trace_rows():
    rows = [
        {"spanID": "s1", "parentSpanID": "", "serviceName": "frontend"},
        {"spanID": "s2", "parentSpanID": "s1", "serviceName": "checkout"},
        {"spanID": "s3", "parentSpanID": "s2", "serviceName": "emailservice"},
    ]
    graph = build_service_graph_from_trace_rows(rows)
    assert graph.ancestors("emailservice") == {"checkout", "frontend"}

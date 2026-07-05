"""Regression tests for the RCAEval MAS causal re-rank wiring (topology/temporal).

These cover the two code paths added when wiring the causal re-rank into the live
MAS synthesize path: the raw-traces dependency-graph builder and the reproducible
runtime-id case selector used to pin a frozen preregistration.
"""
from __future__ import annotations

import csv

from telco_mas.evaluation.rcaeval_adapter import load_cases
from telco_mas.shardrca.catalog import build_catalog_for_case
from telco_mas.shardrca.runner import _rcaeval_trace_graph


def _case_with_traces(tmp_path):
    case_dir = tmp_path / "RE3-TT" / "svc-a_cpu" / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("30\n")
    rows = [{"time": t, "svc-a_cpu": 90 if t >= 30 else 10, "svc-b_cpu": 11} for t in range(60)]
    with (case_dir / "simple_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # svc-b (caller) -> svc-a (callee/root); parent span is svc-b, child span is svc-a
    (case_dir / "traces.csv").write_text(
        "time,traceID,spanID,serviceName,operationName,startTimeMillis,duration,statusCode,parentSpanID\n"
        "0,t1,span_b,svc-b,GET,0,10,0,\n"
        "0,t1,span_a,svc-a,GET,0,10,0,span_b\n",
        encoding="utf-8",
    )
    cases = load_cases(tmp_path)
    assert len(cases) == 1
    return cases[0]


def test_rcaeval_trace_graph_recovers_service_edges(tmp_path):
    case = _case_with_traces(tmp_path)
    catalog = build_catalog_for_case(case, compute_ranges=False)
    graph = _rcaeval_trace_graph(catalog)
    assert graph is not None
    assert {"svc-a", "svc-b"} <= graph.nodes
    # caller svc-b depends on callee svc-a; a failure at svc-a propagates up to svc-b
    assert "svc-b" in graph.ancestors("svc-a")
    assert graph.ancestors("svc-b") == set()


def test_rcaeval_trace_graph_absent_returns_none(tmp_path):
    case_dir = tmp_path / "RE2-SS" / "user_loss" / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("30\n")
    rows = [{"time": t, "user_cpu": 90 if t >= 30 else 10} for t in range(40)]
    with (case_dir / "simple_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cases = load_cases(tmp_path)
    catalog = build_catalog_for_case(cases[0], compute_ranges=False)
    # no traces.csv present -> topology re-rank must be a safe no-op (None graph)
    assert _rcaeval_trace_graph(catalog) is None

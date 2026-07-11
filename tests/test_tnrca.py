from __future__ import annotations

import json

from telco_mas.llm import LLMClient
from telco_mas.tnrca import (
    TNRCADataset,
    audit_graph_leakage,
    evaluate_predictions,
    run_multi_agent,
    run_single_agent,
    sanitize_runtime_graph,
)


def _write_case(root, *, marker: bool = True):
    graph = {
        "nodes": [
            {"@rid": "r1", "@class": "RootCause", "title": "fiber fault"},
            {"@rid": "a1", "@class": "AlarmDetail", "title": "RRU link down", "label": "targetAlarm"},
        ],
        "edges": [
            {"out": "a1", "in": "r1", "@class": "causedBy", **({"label": "targetRootCause"} if marker else {})}
        ],
    }
    label = {"nodes": [{"label": "RootCause", "properties": {"causeName": "fiber fault"}}]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "input.json").write_text(json.dumps(graph), encoding="utf-8")
    (root / "label.json").write_text(json.dumps(label), encoding="utf-8")


def test_leakage_marker_is_detected_and_removed(tmp_path):
    _write_case(tmp_path)
    graph = json.loads((tmp_path / "input.json").read_text())
    findings = audit_graph_leakage(graph)
    assert [(item.path, item.kind) for item in findings] == [("edges[0].label", "answer_marker")]
    clean = sanitize_runtime_graph(graph)
    assert "label" not in clean["edges"][0]
    assert clean["nodes"][1]["label"] == "targetAlarm"
    assert not audit_graph_leakage(clean)


def test_dataset_readiness_refuses_one_leaking_public_example(tmp_path):
    _write_case(tmp_path)
    dataset = TNRCADataset(tmp_path)
    report = dataset.readiness(minimum_confirmatory_cases=2)
    assert len(dataset) == 1
    assert report["raw_cases_with_answer_markers"] == 1
    assert report["raw_protocol_label_safe"] is False
    assert report["confirmatory_ready"] is False
    assert report["sota_comparable"] is False


def test_set_evaluator_uses_macro_case_f1(tmp_path):
    _write_case(tmp_path, marker=False)
    dataset = TNRCADataset(tmp_path)
    result = evaluate_predictions(dataset.cases, {tmp_path.name: ["fiber fault"]})
    assert result["macro_f1"] == 1.0
    assert result["exact_accuracy"] == 1.0


def test_call_matched_runners_use_sanitized_graph(tmp_path):
    _write_case(tmp_path)
    case = TNRCADataset(tmp_path).cases[0]
    calls = []

    def responder(messages, tools):
        serialized = json.dumps(messages)
        assert "targetRootCause" not in serialized
        calls.append(messages[0]["content"])
        return {
            "content": json.dumps(
                {"root_causes": [{"cause_description": "fiber fault", "evidence": ["a1->r1"]}]}
            )
        }

    llm = LLMClient(responder=responder, cache_enabled=False)
    single = run_single_agent(case, llm)
    multi = run_multi_agent(case, llm)
    assert single.root_causes == ("fiber fault",)
    assert multi.root_causes == ("fiber fault",)
    assert single.artifacts["call_budget"] == multi.artifacts["call_budget"] == 4
    assert len(calls) == 8

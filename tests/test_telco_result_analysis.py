import json

from telco_mas.evaluation.telco_result_analysis import analyze_telco_results, main
from telco_mas.synthetic_telco.dataset import build_dataset
from telco_mas.synthetic_telco.prereg import build_preregistration


def test_telco_result_analysis_reports_negative_clear_win_gate(tmp_path):
    result_path = tmp_path / "telco_result.json"
    result_path.write_text(
        json.dumps({
            "meta": {"suite": ["telco_v3"], "systems": ["full", "single"]},
            "summary": {},
            "rows": [
                _row("case_a", "single", True, 10),
                _row("case_a", "full", False, 30),
                _row("case_b", "single", False, 12),
                _row("case_b", "full", True, 36),
            ],
        }),
        encoding="utf-8",
    )

    analysis = analyze_telco_results(result_path)

    assert analysis["benchmark"] == "synthetic_telco"
    assert analysis["clear_win_gate"]["passed"] is False
    assert analysis["paired_metrics"]["diagnosis_correct"]["mcnemar"]["paired_cases"] == 2
    assert analysis["usage"]["treatment_to_baseline_token_ratio"] == 3.0
    assert len(analysis["disagreements"]) == 2


def test_telco_result_analysis_cli_writes_json(tmp_path):
    result_path = tmp_path / "telco_result.json"
    out_path = tmp_path / "analysis.json"
    result_path.write_text(
        json.dumps({
            "meta": {"suite": ["telco_v3"], "systems": ["full", "single"]},
            "summary": {},
            "rows": [_row("case_a", "single", False, 10), _row("case_a", "full", True, 20)],
        }),
        encoding="utf-8",
    )

    rc = main([str(result_path), "--out", str(out_path)])

    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == str(result_path)
    assert payload["primary_metric"] == "diagnosis_correct"


def test_telco_v4_analysis_enforces_complete_frozen_result_grid(tmp_path):
    dataset_path = tmp_path / "v4.json"
    prereg_path = tmp_path / "v4_prereg.json"
    result_path = tmp_path / "v4_result.json"
    dataset = build_dataset(
        suite="telco_v4",
        seed=20260703,
        seed_source="unit-test-fixed",
    )
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    prereg = build_preregistration(
        dataset_path,
        systems=["full", "single"],
        model="test-model",
        temperature=0.0,
        runs=3,
        algorithm_id="git:test",
    )
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    labels = {
        case["labels"]["source_scenario_id"]: case["labels"]
        for case in dataset["cases"]
    }
    rows = []
    for scenario in prereg["row_selection"]["source_scenario_ids"]:
        for system in prereg["systems"]:
            for run in range(3):
                row = _row(scenario, system, system == "full", 20 if system == "full" else 10)
                row["run"] = run
                row["true_fault_type"] = labels[scenario]["fault_type"]
                row["stress_tags"] = labels[scenario]["stress_tags"]
                rows.append(row)
    payload = {
        "meta": {
            "suite": ["telco_v4"],
            "systems": prereg["systems"],
            "scenarios": prereg["row_selection"]["source_scenario_ids"],
            "runs": 3,
            "cache": False,
            "model": "test-model",
            "base_url": prereg["model"]["base_url"],
            "temperature": 0.0,
            "max_tool_iters": prereg["model"]["max_tool_iters"],
            "algorithm_id": "git:test",
            "preregistration": str(prereg_path),
            "frozen_dataset": {
                "path": str(dataset_path),
                "sha256": prereg["dataset"]["sha256"],
                "content_sha256": prereg["dataset"]["content_sha256"],
                "design": prereg["dataset"]["design"],
            },
        },
        "summary": {},
        "rows": rows,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    analysis = analyze_telco_results(result_path)

    assert analysis["confirmatory_integrity"]["passed"] is True
    assert analysis["clear_win_gate"]["integrity_gate_passed"] is True
    assert len(analysis["stratified_diagnostics"]["fault_family"]) == 14
    assert set(
        item["paired_scenarios"]
        for item in analysis["stratified_diagnostics"]["nuisance_profile"].values()
    ) == {14}

    payload["rows"].pop()
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    incomplete = analyze_telco_results(result_path)
    assert incomplete["confirmatory_integrity"]["passed"] is False
    assert (
        incomplete["confirmatory_integrity"]["checks"]["result_rows_complete_and_unique"]
        is False
    )


def _row(case_id: str, system: str, correct: bool, tokens: int) -> dict:
    return {
        "scenario": case_id,
        "system": system,
        "diagnosis_correct": correct,
        "end_to_end_correct": correct,
        "localization": correct,
        "fault_type_correct": correct,
        "causal_explanation_correct": correct,
        "resolved": correct,
        "true_element": "ROOT",
        "predicted_element": "ROOT" if correct else "OTHER",
        "predicted_fault_type": "FAULT",
        "keyword_recall": 1.0 if correct else 0.0,
        "total_tokens": tokens,
        "llm_calls": 1,
        "tool_calls": 1,
        "latency_s": 1.0,
    }

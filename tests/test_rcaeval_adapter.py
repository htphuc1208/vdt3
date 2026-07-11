import csv
import json
from pathlib import Path

import pytest

from telco_mas.evaluation.external import ExternalBenchmarkCase, ExternalPrediction
from telco_mas.evaluation.rcaeval_adapter import (
    apply_rcaeval_metric_prior,
    load_cases,
    metric_prior_for_case,
    score_predictions,
    heuristic_predict,
    validate_rcaeval,
)
from telco_mas.evaluation.rcaeval_paper_comparison import compare


def test_real_rcaeval_symlink_validates_735_cases_if_present():
    validation = validate_rcaeval()
    if not Path(validation["root"]).exists():
        pytest.skip("RCAEval data not present")
    assert validation["ok"] is True
    assert validation["counts"] == {"RE1": 375, "RE2": 270, "RE3": 90}


def test_rcaeval_inference_payload_does_not_leak_labels():
    cases = load_cases(sample=3, seed=3)
    assert cases
    for case in cases:
        payload = case.inference_payload()
        assert "ground_truth_root" not in payload
        assert "fault_type" not in payload
        assert "label_extras" not in payload
        assert case.ground_truth_root not in payload["case_id"]
        assert case.fault_type not in payload["case_id"]
        assert case.fault_type not in payload["tags"]


def test_rcaeval_fixture_metric_case_scores(tmp_path):
    case_dir = tmp_path / "RE1-OB" / "adservice_cpu" / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("30\n")
    rows = []
    for t in range(60):
        rows.append({
            "time": t,
            "adservice_cpu": 90 if t >= 30 else 10,
            "cartservice_cpu": 12,
            "frontend_latency-50": 999,
        })
    with (case_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases = load_cases(tmp_path)
    assert len(cases) == 1
    prediction = heuristic_predict(cases[0])
    scored = score_predictions(cases, [prediction])
    row = scored["rows"][0]
    assert row["hit_at_1"] is True
    assert row["fault_accuracy"] is True


def test_rcaeval_observability_does_not_fallback_to_hidden_fault_label(tmp_path):
    case_dir = tmp_path / "RE1-OB" / "adservice_loss" / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("30\n")
    rows = []
    for t in range(60):
        rows.append({
            "time": t,
            "adservice_queue": 90 if t >= 30 else 10,
            "cartservice_queue": 12,
        })
    with (case_dir / "data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    case = load_cases(tmp_path)[0]

    assert case.fault_type == "loss"
    assert case.observability["likely_fault_family"] == "unknown"
    assert "loss" not in json.dumps(case.inference_payload())


def test_rcaeval_metric_prior_overrides_weak_mas_when_margin_is_strong():
    case = ExternalBenchmarkCase(
        case_id="case-1",
        source="RE2-TT",
        instruction="diagnose",
        ground_truth_root="ts-auth-service",
        fault_type="cpu",
        observability={
            "top_metric_shifts": [
                {"service": "ts-auth-service", "metric": "cpu", "score": 10.0},
                {"service": "ts-order-service", "metric": "cpu", "score": 1.0},
            ],
        },
    )
    prediction = ExternalPrediction(
        case_id="case-1",
        system="rcaeval_shardrca_full",
        root="ts-order-service",
        ranked_roots=["ts-order-service", "ts-auth-service"],
        fault_type="db",
        confidence=0.4,
    )

    repaired = apply_rcaeval_metric_prior(case, prediction)

    assert metric_prior_for_case(case)["strong"] is True
    assert repaired.root == "ts-auth-service"
    assert repaired.ranked_roots[0] == "ts-auth-service"
    assert repaired.fault_type == "cpu"


def test_rcaeval_metric_prior_keeps_mas_primary_when_metric_margin_is_diffuse():
    case = ExternalBenchmarkCase(
        case_id="case-1",
        source="RE2-TT",
        instruction="diagnose",
        ground_truth_root="ts-order-service",
        fault_type="mem",
        observability={
            "top_metric_shifts": [
                {"service": "ts-basic-service", "metric": "mem", "score": 10.0},
                {"service": "ts-route-service", "metric": "mem", "score": 9.5},
                {"service": "ts-order-service", "metric": "mem", "score": 8.0},
            ],
        },
    )
    prediction = ExternalPrediction(
        case_id="case-1",
        system="rcaeval_shardrca_full",
        root="ts-order-service",
        ranked_roots=["ts-order-service"],
        fault_type="mem",
        confidence=0.6,
    )

    repaired = apply_rcaeval_metric_prior(case, prediction)

    assert metric_prior_for_case(case)["strong"] is False
    assert repaired.root == "ts-order-service"
    assert repaired.ranked_roots[:3] == [
        "ts-order-service",
        "ts-basic-service",
        "ts-route-service",
    ]


def test_rcaeval_scoring_treats_final_root_as_primary_prediction():
    case = ExternalBenchmarkCase(
        case_id="case-1",
        source="RE2-TT",
        instruction="diagnose",
        ground_truth_root="ts-auth-service",
        fault_type="cpu",
    )
    prediction = ExternalPrediction(
        case_id="case-1",
        system="rcaeval_shardrca_full",
        root="ts-auth-service",
        ranked_roots=["ts-route-service", "ts-auth-service"],
        fault_type="cpu",
    )

    row = score_predictions([case], [prediction])["rows"][0]

    assert row["hit_at_1"] is True
    assert row["mrr"] == 1.0
    assert row["ranked_roots"][0] == "ts-auth-service"


def test_rcaeval_paper_comparison_pairs_official_baro_with_shardrca(tmp_path):
    case_id = "RCAEval-RE2-TT-ts-auth-service_cpu-1"
    shard_result = tmp_path / "shard.json"
    shard_result.write_text(json.dumps({
        "rows": [{
            "case_id": case_id,
            "true_root": "ts-auth-service",
        }],
    }))
    checkpoints = tmp_path / "checkpoints" / "shardrca_full"
    checkpoints.mkdir(parents=True)
    (checkpoints / f"{case_id}.json").write_text(json.dumps({
        "root": "ts-auth-service",
        "ranked_roots": ["ts-route-service", "ts-auth-service"],
        "total_tokens": 100,
        "llm_calls": 2,
        "tool_calls": 3,
        "latency_s": 4.0,
    }))
    baro = tmp_path / "baro"
    baro.mkdir()
    (baro / "ts-auth-service_cpu_1.json").write_text(json.dumps({
        "0": ["ts-route-service_cpu", "ts-auth-service_cpu"],
    }))

    payload = compare(shard_result, tmp_path / "checkpoints", baro)

    assert payload["meta"]["paired_cases"] == 1
    assert payload["summary"]["rcaeval_shardrca_full"]["ac_at_1"] == 1.0
    assert payload["summary"]["rcaeval_www25_baro"]["avg_at_5"] == 0.8

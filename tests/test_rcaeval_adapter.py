import csv
from pathlib import Path

import pytest

from telco_mas.evaluation.rcaeval_adapter import (
    load_cases,
    score_predictions,
    heuristic_predict,
    validate_rcaeval,
)


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

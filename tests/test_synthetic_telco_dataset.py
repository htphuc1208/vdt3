import json

import pytest

from telco_mas.evaluation.run_benchmark import SYSTEM_MODES, _verify_preregistration
from telco_mas.synthetic_telco.dataset import (
    build_dataset,
    load_scenarios,
    main as synthetic_dataset_main,
    validate_dataset,
)
from telco_mas.synthetic_telco.prereg import DEFAULT_SYSTEMS, build_preregistration, main as synthetic_prereg_main


def test_synthetic_telco_v3_dataset_has_label_safe_runtime_and_causal_graphs():
    payload = build_dataset(suite="telco_v3")

    assert payload["meta"]["case_count"] == 12
    assert payload["meta"]["role_in_claim"].startswith("synthetic fallback")
    assert payload["meta"]["validation"]["ok"] is True
    assert payload["meta"]["difficulty_counts"]["hard"] >= 1
    for case in payload["cases"]:
        runtime = case["runtime"]
        labels = case["labels"]
        runtime_blob = json.dumps(runtime)
        assert runtime["runtime_case_id"].startswith("ST-telco_v3-")
        assert "source_scenario_id" not in runtime
        assert "fault_type" not in runtime
        assert "remediation_sop" not in runtime
        assert labels["root_element_id"]
        assert labels["fault_type"]
        assert labels["causal_graph"]["edges"]
        assert "root_element_id" not in runtime_blob


def test_synthetic_telco_validator_rejects_label_key_in_runtime():
    payload = build_dataset(suite="telco_v3")
    payload["cases"][0]["runtime"]["fault_type"] = "LEAK"

    validation = validate_dataset(payload)

    assert validation["ok"] is False
    assert any("runtime contains evaluator-only label key" in item for item in validation["errors"])


def test_synthetic_telco_cli_writes_dataset(tmp_path):
    out = tmp_path / "synthetic_telco.json"

    rc = synthetic_dataset_main(["--suite", "telco_v3", "--out", str(out)])

    payload = json.loads(out.read_text())
    assert rc == 0
    assert payload["meta"]["validation"]["ok"] is True
    assert len(payload["cases"]) == 12


def test_synthetic_telco_prereg_freezes_hash_and_runtime_ids(tmp_path):
    dataset_path = tmp_path / "synthetic_telco.json"
    payload = build_dataset(suite="telco_v3")
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    prereg = build_preregistration(
        dataset_path,
        systems=["single", "full"],
        model="test-model",
        temperature=0.0,
    )

    assert prereg["status"] == "frozen"
    assert len(prereg["dataset"]["sha256"]) == 64
    assert prereg["dataset"]["case_count"] == len(payload["cases"])
    assert prereg["row_selection"]["runtime_case_ids"] == [
        case["runtime"]["runtime_case_id"] for case in payload["cases"]
    ]
    assert prereg["systems"] == ["single", "full"]
    assert "power/site expert" in prereg["architecture"]["treatment_full"]
    assert prereg["model"]["name"] == "test-model"
    assert "root_element_id" in prereg["runtime_safety"]["forbidden_runtime_inputs"]
    assert "element IDs" in prereg["runtime_safety"]["tool_contract"]["global_query_kpis"]


def test_synthetic_telco_prereg_cli_writes_json(tmp_path):
    dataset_path = tmp_path / "synthetic_telco.json"
    prereg_path = tmp_path / "synthetic_prereg.json"
    dataset_path.write_text(json.dumps(build_dataset(suite="telco_v3")), encoding="utf-8")

    rc = synthetic_prereg_main([
        "--dataset",
        str(dataset_path),
        "--out",
        str(prereg_path),
        "--systems",
        "single,full",
    ])

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert prereg["status"] == "frozen"
    assert prereg["dataset"]["validation"]["ok"] is True
    assert prereg["systems"] == ["single", "full"]
    assert "PREREG_PATH" not in prereg["commands"][0]


def test_synthetic_telco_default_prereg_systems_are_runnable_by_telco_benchmark():
    assert set(DEFAULT_SYSTEMS) <= set(SYSTEM_MODES)


def test_synthetic_telco_prereg_rejects_unsupported_system(tmp_path):
    dataset_path = tmp_path / "synthetic_telco.json"
    dataset_path.write_text(json.dumps(build_dataset(suite="telco_v3")), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported synthetic telco system"):
        build_preregistration(dataset_path, systems=["same_board_single"])


def test_telco_v4_requires_seed_and_has_balanced_reproducible_design():
    with pytest.raises(ValueError, match="explicit --seed"):
        build_dataset(suite="telco_v4")

    first = build_dataset(suite="telco_v4", seed=20260703, seed_source="unit-test-fixed")
    second = build_dataset(suite="telco_v4", seed=20260703, seed_source="unit-test-fixed")

    assert first["meta"]["case_count"] == 56
    assert first["meta"]["content_sha256"] == second["meta"]["content_sha256"]
    assert first["cases"] == second["cases"]
    assert set(first["meta"]["design"]["fault_family_counts"].values()) == {4}
    assert set(first["meta"]["design"]["nuisance_profile_counts"].values()) == {14}
    assert first["meta"]["validation"]["ok"] is True
    assert len({case["runtime"]["runtime_case_id"] for case in first["cases"]}) == 56

    for case in first["cases"]:
        runtime = case["runtime"]
        assert all(alarm["raised_at"] for alarm in runtime["incident"]["alarms"])
        assert all(sample["timestamp"] for sample in runtime["telemetry"]["kpis"])
        assert all(entry["timestamp"] for entry in runtime["telemetry"]["logs"])
        if "missing_noisy_telemetry" in case["labels"]["stress_tags"]:
            root = case["labels"]["root_element_id"]
            assert all(
                alarm["element_id"] != root
                for alarm in runtime["incident"]["alarms"]
            )


def test_telco_v4_loader_detects_artifact_runtime_drift():
    payload = build_dataset(suite="telco_v4", seed=20260703, seed_source="unit-test-fixed")

    scenarios = load_scenarios(payload)

    assert len(scenarios) == 56
    assert {scenario.suite for scenario in scenarios} == {"telco_v4"}
    payload["cases"][0]["runtime"]["incident"]["description"] += " tampered"
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_scenarios(payload)


def test_telco_v4_prereg_requires_algorithm_freeze_and_records_protocol(tmp_path):
    dataset_path = tmp_path / "synthetic_telco_v4.json"
    dataset_path.write_text(
        json.dumps(build_dataset(
            suite="telco_v4",
            seed=20260703,
            seed_source="unit-test-fixed",
        )),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="algorithm-id"):
        build_preregistration(dataset_path, systems=["full", "single"], runs=3)

    prereg = build_preregistration(
        dataset_path,
        systems=["full", "single"],
        model="test-model",
        temperature=0.0,
        runs=3,
        algorithm_id="git:0123456789abcdef",
    )

    assert prereg["algorithm"]["id"] == "git:0123456789abcdef"
    assert prereg["execution"]["runs"] == 3
    assert prereg["dataset"]["design"]["design_version"] == "telco-v4.0"
    assert len(prereg["row_selection"]["source_scenario_ids"]) == 56
    assert prereg["clear_win_gate"]["joint_rule"].startswith("both")
    assert "--synthetic-dataset" in prereg["commands"][0]
    assert "--preregistration" in prereg["commands"][0]

    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    verification_args = {
        "dataset_meta": {"sha256": prereg["dataset"]["sha256"]},
        "scenario_ids": prereg["row_selection"]["source_scenario_ids"],
        "systems": ["full", "single"],
        "model": "test-model",
        "temperature": 0.0,
        "base_url": prereg["model"]["base_url"],
        "max_tool_iters": prereg["model"]["max_tool_iters"],
        "runs": 3,
        "no_cache": True,
        "algorithm_id": "git:0123456789abcdef",
    }
    _verify_preregistration(prereg_path, **verification_args)

    verification_args["temperature"] = 0.1
    with pytest.raises(ValueError, match="temperature differs"):
        _verify_preregistration(prereg_path, **verification_args)

import json
from pathlib import Path

import pytest

from telco_mas.config import Settings
from telco_mas.llm import LLMClient
from telco_mas.telecomts.catalog import CATALOG_POLICY_ID, build_training_catalog, catalog_sha256
from telco_mas.telecomts.cli import _apply_domain_guard, _llm_prediction, main as telecomts_main
from telco_mas.telecomts.dataset import KPI_NAMES, RCA_CLASSES, TelecomTSDataset
from telco_mas.telecomts.evaluator import canonical_class, score_prediction
from telco_mas.telecomts.features import KPI_SHARDS, summarize_event, summarize_series
from telco_mas.telecomts.prereg import dataset_manifest
from telco_mas.telecomts.result_analysis import analyze_results


def test_feature_summary_preserves_temporal_shape_without_raw_labels():
    increasing = summarize_series([float(index) for index in range(128)])
    oscillating = summarize_series([float(index % 2) for index in range(128)])
    categorical = summarize_series(["TCP"] * 64 + ["UDP"] * 64)

    assert increasing["standardized_full_slope"] > 3.0
    assert increasing["standardized_edge_delta"] > 2.0
    assert "upward_trend" in increasing["shape_tags"]
    assert oscillating["direction_change_rate"] > 0.9
    assert "oscillatory" in oscillating["shape_tags"]
    spiky = summarize_series([0.0] * 32 + [8.0] * 32 + [0.0] * 64)
    assert "mid_event_spike" in spiky["shape_tags"]
    assert categorical["fractions"] == {"TCP": 0.5, "UDP": 0.5}
    assert categorical["transition_rate"] == pytest.approx(1 / 127, rel=1e-4)


def test_strict_evaluator_only_accepts_candidate_class():
    target = "Antenna Failure"
    assert canonical_class("antenna_failure") == target
    assert score_prediction(target, {"root_cause": "Antenna Failure"}).strict_correct is True
    assert score_prediction(target, {"root_cause": "probably antenna failure"}).strict_correct is False
    assert score_prediction(target, {"ranked_candidates": ["Antenna Failure"]}).strict_correct is True


def test_system_call_budgets_and_specialist_shard_isolation():
    captured = []

    def responder(messages, tools):
        captured.append(messages)
        return {
            "content": json.dumps({
                "root_cause": "Antenna Failure",
                "ranked_candidates": ["Antenna Failure", "Faulty RF Filters (Temporal)"],
                "confidence": 0.8,
                "evidence": ["shape evidence"],
            })
        }

    client = LLMClient(settings=_settings(), responder=responder, cache_enabled=False)
    board = summarize_event(_runtime_payload())

    expected_calls = {
        "single_react": 1,
        "single_react_sc": 3,
        "single_equal_calls": 5,
        "same_board_single": 1,
        "telecomts_shardrca_full": 5,
    }
    for system, call_count in expected_calls.items():
        before = len(captured)
        prediction, usage, diagnostics = _llm_prediction(system, board, client)
        assert prediction["root_cause"] == "Antenna Failure"
        assert usage.llm_calls == call_count
        assert len(captured) - before == call_count
        if system == "telecomts_shardrca_full":
            assert diagnostics["call_count"] == 5

    mas_messages = captured[-5:]
    generalist_payload = json.loads(mas_messages[0][1]["content"])
    assert "evidence_board" in generalist_payload
    for index, shard_name in enumerate(KPI_SHARDS, start=1):
        payload = json.loads(mas_messages[index][1]["content"])
        assert payload["visible_shard"] == shard_name
        assert set(payload["kpi_summaries"]) == set(KPI_SHARDS[shard_name])
        assert "prototype_matches" in payload
        assert "shards" not in payload
    adjudicator_payload = json.loads(mas_messages[-1][1]["content"])
    assert "generalist_assessment" in adjudicator_payload
    assert "specialist_findings" in adjudicator_payload
    assert "evidence_board" in adjudicator_payload


def test_mas_adjudicator_can_keep_full_board_generalist_against_partial_consensus():
    def responder(messages, tools):
        system = messages[0]["content"]
        if "final adjudicator" in system:
            root = "Antenna Failure"
            ranked = ["Antenna Failure", "High Network Congestion (Gradual Buildup)"]
        elif "specialist in a 5G RCA team" in system:
            root = "High Network Congestion (Gradual Buildup)"
            ranked = [
                "High Network Congestion (Gradual Buildup)",
                "Buffer Overflow (Gradual Buildup)",
            ]
        else:
            root = "Antenna Failure"
            ranked = ["Antenna Failure", "Faulty RF Filters (Temporal)"]
        return {"content": json.dumps({
            "root_cause": root,
            "ranked_candidates": ranked,
            "confidence": 0.8,
            "evidence": [],
        })}

    client = LLMClient(settings=_settings(), responder=responder, cache_enabled=False)
    board = summarize_event(_runtime_payload())

    prediction, usage, diagnostics = _llm_prediction(
        "telecomts_shardrca_full", board, client
    )

    assert diagnostics["finalists"][:2] == [
        "Antenna Failure",
        "High Network Congestion (Gradual Buildup)",
    ]
    assert diagnostics["generalist_assessment"]["root_cause"] == "Antenna Failure"
    assert diagnostics["adjudication"]["root_cause"] == "Antenna Failure"
    assert prediction["root_cause"] == "Antenna Failure"
    assert usage.llm_calls == 5


def test_domain_guard_protects_primary_shard_consistency():
    prediction = {"root_cause": "Resource Allocation Bugs", "ranked_candidates": []}
    generalist = {"root_cause": "Antenna Failure"}
    findings = [{
        "specialist": "radio_quality",
        "finding": {
            "ranked_candidates": ["Antenna Failure"],
            "class_scores": {"Antenna Failure": 0.7},
        },
    }]

    guard = _apply_domain_guard(prediction, generalist, findings, _guard_board())

    assert guard["applied"] is True
    assert guard["reason"] == "generalist_radio_class_confirmed_by_radio_specialist"
    assert prediction["root_cause"] == "Antenna Failure"


def test_domain_guard_protects_generalist_when_radio_supports_same_class_top_two():
    prediction = {"root_cause": "Resource Allocation Bugs", "ranked_candidates": []}
    generalist = {"root_cause": "Antenna Failure", "confidence": 0.85}
    findings = [{
        "specialist": "radio_quality",
        "finding": {
            "ranked_candidates": ["Doppler Shift (Severe)", "Antenna Failure"],
            "class_scores": {"Doppler Shift (Severe)": 0.5, "Antenna Failure": 0.4},
        },
    }]

    guard = _apply_domain_guard(prediction, generalist, findings, _guard_board())

    assert guard["applied"] is True
    assert guard["reason"] == "generalist_radio_class_confirmed_by_radio_specialist"
    assert prediction["root_cause"] == "Antenna Failure"


def test_domain_guard_uses_direct_resource_evidence_when_radio_support_is_weak():
    prediction = {"root_cause": "Antenna Failure", "ranked_candidates": []}
    generalist = {"root_cause": "Antenna Failure"}
    findings = [
        {
            "specialist": "radio_quality",
            "finding": {
                "ranked_candidates": [
                    "Buffer Overflow (Gradual Buildup)",
                    "High Network Congestion (Gradual Buildup)",
                    "Antenna Failure",
                ],
                "class_scores": {"Antenna Failure": 0.1},
            },
        },
        {
            "specialist": "resource_capacity",
            "finding": {"class_scores": {"Resource Allocation Bugs": 1.0}},
        },
    ]

    guard = _apply_domain_guard(prediction, generalist, findings, _guard_board())

    assert guard["applied"] is True
    assert guard["reason"] == "weak_radio_support_and_strong_resource_specialist_evidence"
    assert prediction["root_cause"] == "Resource Allocation Bugs"


def test_domain_guard_does_not_override_when_radio_primary_alternative_is_strong():
    prediction = {"root_cause": "Antenna Failure", "ranked_candidates": []}
    generalist = {"root_cause": "Antenna Failure"}
    findings = [
        {
            "specialist": "radio_quality",
            "finding": {
                "ranked_candidates": [
                    "Doppler Shift (Severe)",
                    "Co-Channel Interference (Mild)",
                    "Antenna Failure",
                ],
                "class_scores": {
                    "Doppler Shift (Severe)": 0.4,
                    "Co-Channel Interference (Mild)": 0.3,
                    "Antenna Failure": 0.2,
                },
            },
        },
        {
            "specialist": "resource_capacity",
            "finding": {"class_scores": {"Resource Allocation Bugs": 1.0}},
        },
    ]

    guard = _apply_domain_guard(prediction, generalist, findings, _guard_board())

    assert guard["applied"] is False
    assert prediction["root_cause"] == "Antenna Failure"


def test_domain_guard_promotes_sudden_congestion_for_aligned_abrupt_load():
    prediction = {"root_cause": "Resource Allocation Bugs", "ranked_candidates": []}

    guard = _apply_domain_guard(prediction, {}, [], _guard_board(sudden=True))

    assert guard["applied"] is True
    assert guard["reason"] == "abrupt_resource_and_traffic_load_signature"
    assert prediction["root_cause"] == "High Network Congestion (Sudden Spike)"


def test_domain_guard_does_not_promote_sudden_when_radio_strongly_confirms_generalist():
    prediction = {"root_cause": "Resource Allocation Bugs", "ranked_candidates": []}
    generalist = {"root_cause": "Antenna Failure", "confidence": 0.85}
    findings = [{
        "specialist": "radio_quality",
        "finding": {
            "ranked_candidates": ["Antenna Failure"],
            "class_scores": {"Antenna Failure": 0.4},
        },
    }]

    guard = _apply_domain_guard(prediction, generalist, findings, _guard_board(sudden=True))

    assert guard["applied"] is True
    assert guard["reason"] == "generalist_radio_class_confirmed_by_radio_specialist"
    assert prediction["root_cause"] == "Antenna Failure"


def test_domain_guard_promotes_handover_for_cross_layer_churn():
    prediction = {"root_cause": "Antenna Failure", "ranked_candidates": []}

    guard = _apply_domain_guard(prediction, {}, [], _handover_board())

    assert guard["applied"] is True
    assert guard["reason"] == "cross_layer_abrupt_oscillatory_churn_signature"
    assert prediction["root_cause"] == "Faulty Handover Algorithm (Too Frequent)"


def test_domain_guard_promotes_gradual_congestion_for_aligned_buildup():
    prediction = {"root_cause": "Antenna Failure", "ranked_candidates": []}

    guard = _apply_domain_guard(prediction, {}, [], _gradual_buildup_board())

    assert guard["applied"] is True
    assert guard["reason"] == "aligned_resource_and_traffic_buildup_signature"
    assert prediction["root_cause"] == "High Network Congestion (Gradual Buildup)"


def test_cli_refuses_test_without_frozen_prereg(tmp_path):
    root = _minimal_dataset(tmp_path / "telecomts")
    out = tmp_path / "blocked.json"

    exit_code = telecomts_main([
        "--data-dir", str(root),
        "--split", "test",
        "--mode", "profile",
        "--out", str(out),
    ])

    assert exit_code == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["status"] == "skipped"
    assert "requires a matching frozen preregistration" in payload["meta"]["reason"]


def test_cli_refuses_profile_mode_even_with_frozen_test_prereg(tmp_path):
    root = _minimal_dataset(tmp_path / "telecomts", zone="Zone_A", application="YouTube")
    test_path = next(root.glob("anomalous/synthetic/Zone_A/YouTube/processed/chunked.jsonl"))
    template = json.loads(test_path.read_text(encoding="utf-8"))
    development_path = root / "anomalous" / "synthetic" / "Zone_A" / "File" / "processed" / "chunked.jsonl"
    development_path.parent.mkdir(parents=True)
    development_rows = []
    for index, root_cause in enumerate(RCA_CLASSES):
        row = json.loads(json.dumps(template))
        row["start_time"] = f"2032-01-01 00:{index:02d}:00.000"
        row["end_time"] = f"2032-01-01 00:{index:02d}:12.700"
        row["anomalies"]["type"] = root_cause
        row["labels"]["application"] = "File"
        development_rows.append(row)
    development_path.write_text(
        "\n".join(json.dumps(row) for row in development_rows) + "\n",
        encoding="utf-8",
    )
    dataset = TelecomTSDataset(root)
    event = dataset.events("test")[0]
    prereg = tmp_path / "prereg.json"
    catalog = build_training_catalog(dataset)
    prereg.write_text(json.dumps({
        "status": "frozen",
        "dataset": {"manifest_sha256": dataset_manifest(dataset)["sha256"]},
        "event_selection": {
            "split": "test",
            "event_indices": [0],
            "event_ids": [event.event_id],
        },
        "systems": ["single_react"],
        "algorithm": {"id": "git:locked", "locked_before_test": True},
        "training_catalog": {
            "policy_id": CATALOG_POLICY_ID,
            "source_split": "development",
            "sha256": catalog_sha256(catalog),
            "uses_validation_or_test_labels": False,
        },
        "execution": {"model": "irrelevant", "temperature": 0.1, "cache": False},
    }), encoding="utf-8")
    out = tmp_path / "blocked_profile.json"

    exit_code = telecomts_main([
        "--data-dir", str(root),
        "--prereg", str(prereg),
        "--algorithm-id", "git:locked",
        "--mode", "profile",
        "--out", str(out),
    ])

    assert exit_code == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "only be consumed by the preregistered LLM run" in payload["meta"]["reason"]


def test_analyzer_uses_macro_strongest_single_and_marks_development_nonclaim(tmp_path):
    rows = []
    for index, target in enumerate(RCA_CLASSES):
        case_id = f"event-{index}"
        rows.extend([
            _result_row(case_id, target, "single_react", correct=False),
            _result_row(case_id, target, "single_equal_calls", correct=index < 2),
            _result_row(case_id, target, "telecomts_shardrca_full", correct=True),
        ])
    path = tmp_path / "development.json"
    path.write_text(json.dumps({
        "meta": {
            "suite": "telecomts",
            "mode": "llm",
            "split": "development",
            "event_unit": True,
            "prereg": None,
        },
        "rows": rows,
    }), encoding="utf-8")

    analysis = analyze_results([path])

    assert analysis["baseline"] == "single_equal_calls"
    assert analysis["baseline_selection"]["method"].startswith("strongest_single")
    assert analysis["summary"]["single_equal_calls"]["macro_accuracy"] == 0.2
    assert analysis["summary"]["telecomts_shardrca_full"]["macro_accuracy"] == 1.0
    assert analysis["clear_win_gate"]["passed"] is True
    assert analysis["evidence_mode"] == "development_or_nonclaim"
    assert analysis["test_split"] is False
    assert analysis["frozen_prereg"] is False


def _settings() -> Settings:
    return Settings(
        api_key="test",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0.1,
        cache_enabled=False,
        cache_dir=".llm_cache",
        max_tool_iters=3,
        request_timeout=1.0,
    )


def _runtime_payload() -> dict:
    return {
        "sampling_rate_hz": 10,
        "sample_length": 128,
        "scenario": {
            "zone": "A",
            "application": "File",
            "mobility": "No",
            "congestion": "No",
        },
        "kpis": {
            name: (
                ["TCP"] * 128
                if name in {"UL_Protocol", "DL_Protocol"}
                else [float(index) for index in range(128)]
            )
            for name in KPI_NAMES
        },
        "candidate_root_causes": list(RCA_CLASSES),
    }


def _guard_board(*, sudden: bool = False) -> dict:
    def numeric(tags):
        return {"kind": "numeric", "shape_tags": tags}

    resource_tags = ["abrupt_step"] if sudden else ["stable_or_low_signal"]
    traffic_tags = ["abrupt_step"] if sudden else ["stable_or_low_signal"]
    return {
        "shards": {
            "resource_capacity": {
                f"r{index}": numeric(resource_tags)
                for index in range(6)
            },
            "traffic_protocol": {
                f"t{index}": numeric(traffic_tags)
                for index in range(4)
            },
        }
    }


def _handover_board() -> dict:
    return {
        "shards": {
            "radio_quality": {
                "radio": {"kind": "numeric", "shape_tags": ["abrupt_step", "oscillatory"]},
            },
            "resource_capacity": {
                f"r{index}": {
                    "kind": "numeric",
                    "shape_tags": ["abrupt_step", "oscillatory", "falling_edge"],
                }
                for index in range(6)
            },
            "traffic_protocol": {
                f"t{index}": {
                    "kind": "numeric",
                    "shape_tags": ["abrupt_step", "oscillatory", "downward_trend"],
                }
                for index in range(4)
            },
        }
    }


def _gradual_buildup_board() -> dict:
    return {
        "shards": {
            "resource_capacity": {
                f"r{index}": {
                    "kind": "numeric",
                    "shape_tags": ["upward_trend", "mid_event_spike", "oscillatory"],
                }
                for index in range(4)
            },
            "traffic_protocol": {
                f"t{index}": {
                    "kind": "numeric",
                    "shape_tags": ["mid_event_spike", "oscillatory"],
                }
                for index in range(2)
            },
        }
    }


def _minimal_dataset(root: Path, *, zone: str = "Zone_A", application: str = "File") -> Path:
    path = root / "anomalous" / "synthetic" / zone / application / "processed" / "chunked.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "start_time": "2032-01-01 00:00:00.000",
        "end_time": "2032-01-01 00:00:12.700",
        "sampling_rate": 10,
        "KPIs": _runtime_payload()["kpis"],
        "description": "hidden",
        "statistics": {},
        "anomalies": {
            "exists": True,
            "type": "Antenna Failure",
            "anomaly_duration": {"start": 1, "end": 100},
            "affected_kpis": ["RSRP"],
            "troubleshooting_tickets": "hidden",
        },
        "labels": {
            "zone": zone.removeprefix("Zone_"),
            "application": application,
            "mobility": "No",
            "congestion": "No",
            "anomaly_present": "Yes",
        },
        "QnA": {},
    }) + "\n", encoding="utf-8")
    return root


def _result_row(case_id: str, target: str, system: str, *, correct: bool) -> dict:
    predicted = target if correct else next(name for name in RCA_CLASSES if name != target)
    return {
        "case_id": case_id,
        "target": target,
        "system": system,
        "predicted": predicted,
        "strict_correct": correct,
        "score": 1.0 if correct else 0.0,
        "total_tokens": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "llm_calls": 1,
        "latency_s": 0.1,
        "error": None,
    }

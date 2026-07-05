import json
from pathlib import Path

import pytest

from telco_mas.telecomts.catalog import build_training_catalog, catalog_sha256
from telco_mas.telecomts.dataset import (
    KPI_NAMES,
    RCA_CLASSES,
    TelecomTSDataset,
    TelecomTSDatasetError,
)
from telco_mas.evaluation.benchmark_readiness import check_telecomts
from telco_mas.telecomts.features import summarize_event
from telco_mas.telecomts.prereg import build_preregistration


def test_loader_uses_source_session_latin_square_and_opaque_ids(tmp_path):
    root = _telecomts_fixture(tmp_path / "telecomts")
    dataset = TelecomTSDataset(root)

    assert dataset.complete_official_layout is True
    assert dataset.counts() == {"development": 30, "validation": 30, "test": 30}
    assert dataset.event_counts() == {"development": 30, "validation": 30, "test": 30}
    assert {record.source_session for record in dataset.rows("development")} == {
        "Zone_A/File",
        "Zone_B/Twitch",
        "Zone_C/YouTube",
    }
    assert {record.source_session for record in dataset.rows("validation")} == {
        "Zone_A/Twitch",
        "Zone_B/YouTube",
        "Zone_C/File",
    }
    assert {record.source_session for record in dataset.rows("test")} == {
        "Zone_A/YouTube",
        "Zone_B/File",
        "Zone_C/Twitch",
    }
    assert dataset.class_counts("test") == {name: 3 for name in RCA_CLASSES}
    assert dataset.class_event_counts("test") == {name: 3 for name in RCA_CLASSES}

    task = dataset.get_runtime_task("test", 0)
    assert task["case_id"].startswith("TTS-")
    assert "Zone_" not in task["case_id"]
    assert set(task["payload"]["kpis"]) == set(KPI_NAMES)
    assert task["payload"]["candidate_root_causes"] == list(RCA_CLASSES)
    assert set(task["payload"]) == {
        "sampling_rate_hz",
        "sample_length",
        "scenario",
        "kpis",
        "candidate_root_causes",
    }
    forbidden = {
        "anomalies",
        "affected_kpis",
        "troubleshooting_tickets",
        "qna",
        "description",
        "statistics",
        "start_time",
        "end_time",
        "source_path",
        "anomaly_present",
    }
    assert not (_recursive_keys(task) & forbidden)
    assert dataset.target("test", 0) in RCA_CLASSES

    event = dataset.get_runtime_event("test", 0)
    assert event["case_id"].startswith("TTE-")
    assert event["payload"]["sample_length"] == 128
    assert not (_recursive_keys(event) & forbidden)
    assert dataset.event_target("test", 0) in RCA_CLASSES


def test_loader_rejects_jamming_as_upstream_rca_class(tmp_path):
    root = tmp_path / "telecomts"
    path = root / "anomalous" / "synthetic" / "Zone_A" / "File" / "processed" / "chunked.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_row("Jamming", zone="A", application="File")) + "\n", encoding="utf-8")

    dataset = TelecomTSDataset(root)
    with pytest.raises(TelecomTSDatasetError, match="Unsupported TelecomTS RCA class"):
        dataset.records()


def test_loader_merges_overlapping_windows_into_one_event(tmp_path):
    root = tmp_path / "telecomts"
    path = root / "anomalous" / "synthetic" / "Zone_A" / "File" / "processed" / "chunked.jsonl"
    path.parent.mkdir(parents=True)
    first = _row("Antenna Failure", zone="A", application="File")
    second = _row("Antenna Failure", zone="A", application="File")
    second["start_time"] = "2032-01-01 00:00:03.200"
    second["end_time"] = "2032-01-01 00:00:15.900"
    path.write_text(
        "\n".join([json.dumps(first), json.dumps(second)]) + "\n",
        encoding="utf-8",
    )

    dataset = TelecomTSDataset(root)
    assert len(dataset.rows("development")) == 2
    assert len(dataset.events("development")) == 1
    task = dataset.get_runtime_event("development", 0)
    assert task["payload"]["sample_length"] == 160
    assert all(len(values) == 160 for values in task["payload"]["kpis"].values())


def test_preregistration_is_draft_until_algorithm_is_locked(tmp_path):
    dataset = TelecomTSDataset(_telecomts_fixture(tmp_path / "telecomts"))

    draft = build_preregistration(dataset, limit_per_class=1)
    assert draft["status"] == "draft"
    assert draft["event_selection"]["split"] == "test"
    assert draft["event_selection"]["event_count"] == 10
    assert draft["event_selection"]["class_counts"] == {name: 1 for name in RCA_CLASSES}
    assert draft["split_policy"]["test"] == ["Zone_A/YouTube", "Zone_B/File", "Zone_C/Twitch"]
    assert draft["benchmark"]["evidence_tier"] == "public_5g_testbed_backed_synthetic_rca"
    assert draft["algorithm"]["locked_before_test"] is False

    with pytest.raises(ValueError, match="algorithm-id"):
        build_preregistration(dataset, limit_per_class=1, freeze=True)

    frozen = build_preregistration(
        dataset,
        limit_per_class=1,
        freeze=True,
        algorithm_id="git:0123456789abcdef",
    )
    assert frozen["status"] == "frozen"
    assert frozen["algorithm"]["id"] == "git:0123456789abcdef"
    assert frozen["stopping_rule"]["test_used_once_after_algorithm_lock"] is True


def test_training_catalog_uses_development_only_and_is_frozen_in_prereg(tmp_path):
    dataset = TelecomTSDataset(_telecomts_fixture(tmp_path / "telecomts"))
    catalog = build_training_catalog(dataset)

    assert set(catalog) == set(RCA_CLASSES)
    assert catalog["Antenna Failure"]["affected_kpis"] == ["RSRP"]
    assert catalog["Antenna Failure"]["training_support_windows"] == 3
    assert catalog["Antenna Failure"]["prototype_support_events"] == 3
    assert "radio_quality" in catalog["Antenna Failure"]["shape_prototype"]
    assert "RSRP" in catalog["Antenna Failure"]["shape_prototype"]["radio_quality"]
    with pytest.raises(ValueError, match="development split"):
        build_training_catalog(dataset, split="validation")

    task = dataset.get_runtime_event("development", 0)
    board = summarize_event(
        task["payload"],
        class_catalog=catalog,
        include_prototype_matches=True,
    )
    assert len(board["prototype_matches"]["overall"]) == len(RCA_CLASSES)
    assert "shape_prototype" not in board["class_catalog"]["Antenna Failure"]
    assert set(board["prototype_matches"]["by_shard"]) == {
        "radio_quality",
        "resource_capacity",
        "traffic_protocol",
    }
    assert board["prototype_matches"]["overall"][0]["class"] in RCA_CLASSES

    prereg = build_preregistration(dataset, limit_per_class=1)
    assert prereg["training_catalog"]["source_split"] == "development"
    assert prereg["training_catalog"]["uses_validation_or_test_labels"] is False
    assert prereg["training_catalog"]["sha256"] == catalog_sha256(catalog)


def test_readiness_requires_frozen_prereg_before_synthetic_evaluation(tmp_path):
    root = _telecomts_fixture(tmp_path / "telecomts")
    dataset = TelecomTSDataset(root)
    prereg_path = tmp_path / "prereg.json"

    missing = check_telecomts(root, prereg_path)
    assert missing["status"] == "needs_frozen_prereg"
    assert missing["ready_for_synthetic_fallback"] is False
    assert missing["current"]["complete_official_layout"] is True

    prereg_path.write_text(json.dumps(build_preregistration(
        dataset,
        limit_per_class=1,
        freeze=True,
        algorithm_id="git:0123456789abcdef",
    )), encoding="utf-8")
    frozen = check_telecomts(root, prereg_path)
    assert frozen["prereg"]["matches"] is True
    assert frozen["status"] == "ready_synthetic_fallback"
    assert frozen["local_support"]["runner"] is True
    assert frozen["ready_for_synthetic_fallback"] is True

    first = dataset.jsonl_paths()[0]
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    stale = check_telecomts(root, prereg_path)
    assert stale["status"] == "needs_frozen_prereg"
    assert stale["prereg"]["manifest_matches"] is False


def _telecomts_fixture(root: Path) -> Path:
    for zone in ("Zone_A", "Zone_B", "Zone_C"):
        for application in ("File", "Twitch", "YouTube"):
            path = root / "anomalous" / "synthetic" / zone / application / "processed" / "chunked.jsonl"
            path.parent.mkdir(parents=True)
            rows = [
                _row(name, zone=zone.removeprefix("Zone_"), application=application)
                for name in RCA_CLASSES
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return root


def _row(root_cause: str, *, zone: str, application: str) -> dict:
    kpis = {
        name: (["TCP"] * 128 if name in {"UL_Protocol", "DL_Protocol"} else [0.0] * 128)
        for name in KPI_NAMES
    }
    return {
        "start_time": "2032-01-01 00:00:00.000",
        "end_time": "2032-01-01 00:00:12.700",
        "sampling_rate": 10,
        "KPIs": kpis,
        "description": f"label-bearing generated description for {root_cause}",
        "statistics": {"hidden": True},
        "anomalies": {
            "exists": True,
            "type": root_cause,
            "anomaly_duration": {"start": 2, "end": 100},
            "affected_kpis": ["RSRP"],
            "troubleshooting_tickets": f"Root Cause: {root_cause}",
        },
        "labels": {
            "zone": zone,
            "application": application,
            "mobility": "No",
            "congestion": "No",
            "anomaly_present": "Yes",
        },
        "QnA": {
            "anomalies": [{"q": "root?", "a": root_cause, "reasoning": root_cause}],
        },
    }


def _recursive_keys(value) -> set[str]:
    if isinstance(value, dict):
        out = {str(key).lower() for key in value}
        for child in value.values():
            out.update(_recursive_keys(child))
        return out
    if isinstance(value, list):
        out = set()
        for child in value:
            out.update(_recursive_keys(child))
        return out
    return set()

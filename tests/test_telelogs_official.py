import json
from pathlib import Path

from telco_mas.llm import LLMClient
from telco_mas.telelogs import cli as telelogs_cli
from telco_mas.telelogs.dataset import TeleLogsDataset
from telco_mas.telelogs.evaluator import evaluator_label_set, score_prediction
from telco_mas.telelogs.prereg import build_preregistration
from telco_mas.telelogs.result_analysis import analyze_results


def test_telelogs_runtime_task_strips_root_cause_labels(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = TeleLogsDataset(root)

    task = dataset.get_runtime_task("test", 0)
    payload_blob = json.dumps(task["payload"])

    assert dataset.counts() == {"test": 2, "train": 1}
    assert task["task_id"] == "tl-test-1"
    assert "throughput_mbps" in payload_blob
    assert "root_causes" not in payload_blob
    assert "pci mod 30 collision" not in payload_blob


def test_telelogs_evaluator_requires_structured_exact_set(tmp_path):
    root = _fixture_dataset(tmp_path)
    row = TeleLogsDataset(root).rows("test")[0]

    assert evaluator_label_set(row) == {"pci mod 30 collision", "non colocated co frequency interference"}
    exact = score_prediction(row, {"root_causes": ["PCI mod 30 collision", "non-colocated co-frequency interference"]})
    partial = score_prediction(row, {"root_causes": ["PCI mod 30 collision"]})
    prose = score_prediction(row, {"rationale": "Likely PCI mod 30 collision and non-colocated co-frequency interference."})

    assert exact.strict_correct is True
    assert exact.score == 1.0
    assert partial.strict_correct is False
    assert partial.score == 0.5
    assert prose.strict_correct is False
    assert prose.score == 1.0
    assert prose.method == "telelogs_label_text_recall"


def test_telelogs_preregistration_freezes_test_split(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = TeleLogsDataset(root)

    payload = build_preregistration(
        dataset,
        split="test",
        systems=["single_react_sc", "shardrca_full"],
        row_ids=[0, 1],
        model="test-model",
    )

    assert payload["status"] == "frozen"
    assert payload["dataset"]["counts"] == {"test": 2, "train": 1}
    assert payload["row_selection"]["split"] == "test"
    assert payload["row_selection"]["row_ids"] == [0, 1]
    assert payload["dataset"]["manifest_sha256"]
    assert "root_causes" in payload["runtime_safety"]["label_like_keys_removed_from_payload"]
    assert "--baseline strongest_single" in payload["analysis_command"]


def test_telelogs_profile_cli_and_analysis(tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    out = tmp_path / "profile.json"

    rc = telelogs_cli.main([
        "--data-dir", str(root),
        "--split", "test",
        "--systems", "single_react_sc,shardrca_full",
        "--limit", "1",
        "--mode", "profile",
        "--out", str(out),
    ])

    payload = json.loads(out.read_text(encoding="utf-8"))
    analysis = analyze_results([out], baseline="strongest_single")
    assert rc == 0
    assert payload["meta"]["status"] == "completed"
    assert len(payload["rows"]) == 2
    assert analysis["benchmark"] == "telelogs"
    assert analysis["paired"]["strict_correct"]["mcnemar"]["paired_cases"] == 1
    assert analysis["clear_win_gate"]["passed"] is False


def test_telelogs_llm_cli_with_stubbed_single_and_mas(monkeypatch, tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    dataset = TeleLogsDataset(root)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(build_preregistration(
        dataset,
        split="test",
        systems=["single_react_sc", "shardrca_full"],
        row_ids=[0],
    )), encoding="utf-8")
    out = tmp_path / "llm.json"

    def responder(messages, tools):
        system_prompt = messages[0]["content"]
        if "synthesis agent" in system_prompt:
            content = {
                "root_causes": ["PCI mod 30 collision", "non-colocated co-frequency interference"],
                "confidence": 0.9,
                "rationale": "role agreement",
            }
        elif "one careful" in system_prompt:
            content = {
                "root_causes": ["PCI mod 30 collision"],
                "confidence": 0.6,
                "rationale": "single missed one co-channel cause",
            }
        else:
            content = {
                "root_causes": ["PCI mod 30 collision"],
                "confidence": 0.5,
                "evidence": ["stub"],
                "rationale": "role finding",
            }
        return {"content": json.dumps(content), "tool_calls": []}

    monkeypatch.setattr(
        telelogs_cli,
        "LLMClient",
        lambda *args, **kwargs: LLMClient(responder=responder, cache_enabled=False),
    )
    monkeypatch.setattr(telelogs_cli, "get_settings", lambda: type("S", (), {"has_api_key": True})())

    rc = telelogs_cli.main([
        "--data-dir", str(root),
        "--prereg", str(prereg_path),
        "--mode", "llm",
        "--out", str(out),
        "--no-cache",
    ])

    payload = json.loads(out.read_text(encoding="utf-8"))
    analysis = analyze_results([out])
    by_system = {row["system"]: row for row in payload["rows"]}
    assert rc == 0
    assert by_system["single_react_sc"]["strict_correct"] is False
    assert by_system["shardrca_full"]["strict_correct"] is True
    assert by_system["shardrca_full"]["llm_calls"] == 4
    assert analysis["baseline"] == "single_react_sc"
    assert analysis["baseline_selection"]["resolved"] == "single_react_sc"
    assert analysis["paired"]["strict_correct"]["mcnemar"]["treatment_only_correct"] == 1


def _fixture_dataset(root: Path) -> Path:
    folder = root / "troubleshooting"
    folder.mkdir(parents=True)
    train_rows = [
        {
            "id": "tl-train-1",
            "question": "Diagnose throughput degradation.",
            "network": {"speed_kmh": 50, "throughput_mbps": 420},
            "root_causes": ["vehicle speed exceeds 40 km/h"],
        }
    ]
    test_rows = [
        {
            "id": "tl-test-1",
            "question": "Diagnose 5G throughput degradation.",
            "network": {
                "throughput_mbps": 350,
                "serving_cell_rsrp": -112,
                "serving_cell_sinr": 3,
                "neighbor_pci_mod_30": "same",
            },
            "root_causes": ["PCI mod 30 collision", "non-colocated co-frequency interference"],
        },
        {
            "id": "tl-test-2",
            "question": "Diagnose weak far-end coverage.",
            "network": {"coverage_distance_km": 1.4, "downtilt_deg": 12},
            "root_cause": "serving cell coverage distance exceeds 1 km",
        },
    ]
    (folder / "train.json").write_text(json.dumps(train_rows), encoding="utf-8")
    (folder / "test.json").write_text(json.dumps(test_rows), encoding="utf-8")
    return root

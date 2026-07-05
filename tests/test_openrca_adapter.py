import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telco_mas.openrca.dataset import OpenRCADataset
from telco_mas.openrca.error_analysis import classify_error
from telco_mas.openrca.evaluator import evaluate_prediction
from telco_mas.openrca.fit_repair_weights import load_cases as load_openrca_fit_cases
from telco_mas.openrca.formatter import format_prediction
from telco_mas.openrca.prereg import build_preregistration
from telco_mas.openrca.prepared import PreparedOpenRCA, prepare_dataset
from telco_mas.openrca.rca_agent_replica import run_rca_agent_replica
from telco_mas.openrca.result_analysis import analyze_openrca_results
from telco_mas.openrca.sandbox_kernel import validate_code
from telco_mas.openrca.schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput
from telco_mas.openrca.task_parser import parse_runtime_task
from telco_mas.openrca.tools import OpenRCATelemetryTools
from telco_mas.openrca.workers import targeted_falsify
from telco_mas.schemas import UsageStats
from telco_mas.shardrca.board import Blackboard, CandidateRootCause, Finding
from telco_mas.shardrca.runner import (
    ShardRCARunResult,
    _openrca_occurrence_value,
    _openrca_output_component,
    run_openrca_task,
)
from telco_mas.shardrca.synthesizer import SynthesizerResult
from telco_mas.llm import LLMClient


def test_openrca_runtime_task_does_not_leak_scoring_points(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = OpenRCADataset(root)
    task = dataset.get_runtime_task(0)
    assert task["row_id"] == 0
    assert task["task_index"] == "task_7"
    assert "instruction" in task
    assert "scoring_points" not in task
    assert "record" not in task
    assert "docker_001" in dataset.get_scoring_points(0)


def test_openrca_formatter_and_evaluator_accept_official_shape():
    prediction = format_prediction(
        OpenRCAPredictionOutput(
            root_causes=[
                OpenRCAPredictionItem(
                    root_cause_occurrence_datetime="2020-04-11 00:15:30",
                    root_cause_component="docker_001",
                    root_cause_reason="CPU fault",
                )
            ]
        )
    )
    scoring_points = "\n".join([
        "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2020-04-11 00:15:00",
        "The only predicted root cause component is docker_001",
        "The only predicted root cause reason is CPU fault",
    ])
    passed, failed, score = evaluate_prediction(prediction, scoring_points)
    assert score == 1.0
    assert failed == []
    assert "docker_001" in passed


def test_openrca_metric_tool_returns_bounded_anomaly_summary(tmp_path):
    root = _fixture_dataset(tmp_path)
    metric_path = root / "Telecom" / "telemetry" / "2020_04_11" / "metric" / "metric_container.csv"
    metric_path.write_text(
        "\n".join([
            "itemid,name,bomc_id,timestamp,value,cmdb_id",
            f"1,cpu_usage,ZJ,{_ms('2020-04-11 00:00:00')},1.0,docker_001",
            f"2,cpu_usage,ZJ,{_ms('2020-04-11 00:05:00')},1.1,docker_001",
            f"3,cpu_usage,ZJ,{_ms('2020-04-11 00:10:00')},1.2,docker_001",
            f"4,cpu_usage,ZJ,{_ms('2020-04-11 00:15:00')},50.0,docker_001",
        ]),
        encoding="utf-8",
    )
    dataset = OpenRCADataset(root)
    tools = OpenRCATelemetryTools(dataset, max_rows_per_file=20)
    summary = tools.summarize_metric_anomalies(
        "2020-04-11 00:15:00",
        "2020-04-11 00:16:00",
        metric_file="metric_container.csv",
        components=["docker_001"],
        limit=3,
    )
    assert summary["files_scanned"] == 1
    assert len(summary["anomalies"]) <= 3
    assert summary["anomalies"][0]["component"] == "docker_001"
    assert summary["anomalies"][0]["metric"] == "cpu_usage"
    assert summary["anomalies"][0]["direction"] == "high"


def test_openrca_natural_language_task_parser_uses_utc_plus_8_and_task_contract():
    task = {
        "row_id": 7,
        "task_index": "task_7",
        "instruction": (
            "During the specified time range of May 22, 2020, from 00:30 to 01:00, "
            "the system experienced a single failure."
        ),
    }

    parsed = parse_runtime_task(task)

    assert parsed.date_key == "2020_05_22"
    assert parsed.start.strftime("%Y-%m-%d %H:%M:%S %z") == "2020-05-22 00:30:00 +0800"
    assert parsed.end.strftime("%Y-%m-%d %H:%M:%S %z") == "2020-05-22 01:00:00 +0800"
    assert set(parsed.requested_fields) == {
        "root cause occurrence datetime",
        "root cause component",
        "root cause reason",
    }


def test_openrca_time_without_component_uses_early_window_prior():
    board = Blackboard(case_id="t")
    result = ShardRCARunResult(
        system="shardrca_full",
        board=board,
        synthesis=SynthesizerResult(winner=CandidateRootCause(component="UNKNOWN"), candidates=[]),
        winner=CandidateRootCause(component="UNKNOWN"),
        usage=UsageStats(),
        latency_s=0.0,
    )
    start = _ms("2020-04-11 00:00:00")
    end = _ms("2020-04-11 00:30:00")

    value = _openrca_occurrence_value(
        result,
        {"root cause occurrence datetime", "root cause reason"},
        start,
        end,
    )

    assert value == start + (end - start) / 3.0


def test_openrca_component_only_prior_winner_falls_back_to_strongest_finding():
    board = Blackboard(case_id="t")
    board.add(
        Finding(
            shard_id="metrics",
            modality="metrics",
            component="os_021",
            signal="CPU_util_pct",
            score=4.0,
            evidence_ptr="metric#os_021",
        )
    )
    winner = CandidateRootCause(
        component="db_007",
        reason="CPU fault",
        evidence=[],
        rationale="Equal-weight product-of-experts posterior=0.1; explicit_supporters=none.",
    )
    result = ShardRCARunResult(
        system="shardrca_full",
        board=board,
        synthesis=SynthesizerResult(winner=winner, candidates=[winner]),
        winner=winner,
        usage=UsageStats(),
        latency_s=0.0,
        artifacts={"candidate_catalog": {"components": ["db_007", "os_021"]}},
    )

    assert _openrca_output_component(result, {"root cause component"}) == "os_021"
    assert _openrca_output_component(
        result,
        {"root cause component", "root cause reason"},
    ) == "db_007"


def test_prepared_openrca_cache_is_label_safe_and_drives_offline_workers(tmp_path):
    root = _fixture_dataset(tmp_path)
    metric_path = root / "Telecom" / "telemetry" / "2020_04_11" / "metric" / "metric_container.csv"
    metric_path.write_text(
        "\n".join([
            "itemid,name,bomc_id,timestamp,value,cmdb_id",
            f"1,cpu_usage,ZJ,{_ms('2020-04-11 00:00:00')},1.0,docker_001",
            f"2,cpu_usage,ZJ,{_ms('2020-04-11 00:05:00')},1.0,docker_001",
            f"3,cpu_usage,ZJ,{_ms('2020-04-11 00:10:00')},1.0,docker_001",
            f"4,cpu_usage,ZJ,{_ms('2020-04-11 00:15:00')},80.0,docker_001",
        ]),
        encoding="utf-8",
    )
    dataset = OpenRCADataset(root)
    prepared_root = tmp_path / "prepared"
    manifest = prepare_dataset(dataset, prepared_root, chunksize=2)
    prepared = PreparedOpenRCA(prepared_root)

    prediction, result = run_openrca_task(
        dataset,
        dataset.get_runtime_task(0),
        system="shardrca_full",
        llm=None,
        prepared=prepared,
        chunksize=2,
    )

    assert manifest["row_count"] == 1
    assert "scoring_points" not in (prepared.row_dir(0) / "task.json").read_text()
    assert result.artifacts["fusion"] == "correlation_aware_log_opinion_pool"
    assert result.artifacts["candidate_catalog_source"]["components"] == "prepared_runtime_telemetry"
    assert result.artifacts["candidate_catalog_source"]["label_derived"] is False
    # Default weights are a strict no-op: the fused posterior equals equal-weight PoE.
    assert result.artifacts["fusion_weights"]["correlation_rho"] == 0.0
    assert result.artifacts["fusion_weights"]["modality_reliability_enabled"] is False
    assert prediction.root_causes


def test_openrca_heuristic_floor_is_deterministic_and_label_safe(tmp_path):
    root = _fixture_dataset(tmp_path)
    metric_path = root / "Telecom" / "telemetry" / "2020_04_11" / "metric" / "metric_container.csv"
    metric_path.write_text(
        "\n".join([
            "itemid,name,bomc_id,timestamp,value,cmdb_id",
            f"1,cpu_usage,ZJ,{_ms('2020-04-11 00:00:00')},1.0,docker_001",
            f"2,cpu_usage,ZJ,{_ms('2020-04-11 00:15:00')},99.0,docker_001",
        ]),
        encoding="utf-8",
    )
    dataset = OpenRCADataset(root)
    prepared_root = tmp_path / "prepared"
    prepare_dataset(dataset, prepared_root, chunksize=2)
    prepared = PreparedOpenRCA(prepared_root)

    prediction, result = run_openrca_task(
        dataset,
        dataset.get_runtime_task(0),
        system="heuristic_floor",
        llm=None,
        prepared=prepared,
        chunksize=2,
    )

    assert result.system == "heuristic_floor"
    assert result.artifacts["architecture"] == "prepared_telemetry_heuristic_floor"
    assert result.artifacts["candidate_catalog_source"]["label_derived"] is False
    assert "scoring_points" not in (prepared.row_dir(0) / "task.json").read_text()
    assert prediction.root_causes[0].root_cause_component == "docker_001"


def test_targeted_falsifier_promotes_supported_runner_with_evidence():
    board = Blackboard(case_id="t")
    board.add(
        Finding(
            shard_id="s",
            modality="metrics",
            component="runner",
            signal="cpu",
            score=2.0,
            evidence_ptr="metric#runner",
            metadata={"reason_hint": "CPU fault"},
        )
    )
    top = CandidateRootCause(component="top", reason="CPU fault", score=0.6)
    runner = CandidateRootCause(component="runner", reason="CPU fault", score=0.5)

    winner, usage, diagnostic = targeted_falsify(board, [top, runner], llm=None)

    assert winner.component == "runner"
    assert usage.tool_calls == 2
    assert diagnostic["selected"] == "runner_up"
    assert diagnostic["runner_up"]["status"] == "SUPPORTED"
    assert diagnostic["runner_up"]["evidence_ptrs"] == ["metric#runner"]


def test_openrca_trace_edges_feed_topology_artifacts(tmp_path):
    root = _fixture_dataset(tmp_path)
    trace_path = root / "Telecom" / "telemetry" / "2020_04_11" / "trace" / "trace_span.csv"
    trace_path.write_text(
        "\n".join([
            "callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,dsName,serviceName",
            f"http,{_ms('2020-04-11 00:15:00')},10,true,t1,span1,,docker_001,GET,docker_001",
            f"http,{_ms('2020-04-11 00:15:01')},80,true,t1,span2,span1,docker_002,GET,docker_002",
        ]),
        encoding="utf-8",
    )
    dataset = OpenRCADataset(root)
    prepared_root = tmp_path / "prepared"
    prepare_dataset(dataset, prepared_root, chunksize=2)
    prepared = PreparedOpenRCA(prepared_root)

    _, result = run_openrca_task(
        dataset,
        dataset.get_runtime_task(0),
        system="shardrca_full",
        llm=None,
        prepared=prepared,
        chunksize=2,
    )

    assert prepared.trace_edges_path(0) is not None
    assert result.artifacts["topology"]["enabled"] is True
    assert result.artifacts["topology"]["edge_count"] >= 1
    assert result.artifacts["fusion_weights"]["topology_gamma"] == 0.0


def test_openrca_error_taxonomy_classifies_field_failures():
    base = {
        "row_id": 0,
        "task_index": "task_7",
        "strict_correct": False,
        "score": 0.0,
        "volume_bin": "high",
        "scoring_points": "\n".join([
            "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2020-04-11 00:15:00",
            "The only predicted root cause component is docker_001",
            "The only predicted root cause reason is CPU fault",
        ]),
    }
    component = classify_error({**base, "prediction": _prediction("docker_002", "CPU fault", "2020-04-11 00:15:00")})
    reason = classify_error({**base, "prediction": _prediction("docker_001", "network delay", "2020-04-11 00:15:00")})
    time = classify_error({**base, "prediction": _prediction("docker_001", "CPU fault", "2020-04-11 00:20:00")})
    bad_format = classify_error({**base, "prediction": "{bad json"})
    missing = classify_error({**base, "prediction": json.dumps({"1": {"root cause component": "docker_001"}})})
    multi = classify_error({
        **base,
        "prediction": json.dumps({
            "1": {"root cause component": "docker_001", "root cause reason": "CPU fault"},
            "2": {"root cause component": "docker_002", "root cause reason": "CPU fault"},
        }),
    })
    invalid = classify_error({**base, "prediction": _prediction("not_a_component", "CPU fault", "2020-04-11 00:15:00")})

    assert component["categories"] == ["component"]
    assert reason["categories"] == ["reason"]
    assert time["categories"] == ["time"]
    assert "format" in bad_format["categories"]
    assert "missing_field" in missing["categories"]
    assert "multi_root_count" in multi["categories"]
    assert "invalid_catalog" in invalid["categories"]


def test_openrca_repair_fitter_requires_declared_non_forbidden_dev_rows(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "rows": [
            {
                "row_id": 3,
                "system": "shardrca_full",
                "scoring_points": "The only predicted root cause component is docker_001",
                "artifacts": {
                    "candidate_catalog": {
                        "components": ["docker_001"],
                        "reasons": ["CPU fault"],
                    },
                    "worker_distributions": [
                        {
                            "worker_id": "w",
                            "modality": "metrics",
                            "candidate_scope": ["docker_001"],
                            "candidates": [
                                {
                                    "component": "docker_001",
                                    "reason_family": "CPU fault",
                                    "probability": 0.9,
                                    "modality": "metrics",
                                    "worker_id": "w",
                                    "evidence": ["ptr"],
                                }
                            ],
                            "other_mass": 0.1,
                        }
                    ],
                },
            }
        ]
    }), encoding="utf-8")

    cases = load_openrca_fit_cases([result], dev_rows={"3"}, forbidden_rows=set())
    assert len(cases) == 1
    assert cases[0].true_component == "docker_001"
    try:
        load_openrca_fit_cases([result], dev_rows={"3"}, forbidden_rows={"3"})
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("forbidden dev row was accepted")


def test_sandbox_ast_policy_rejects_process_network_and_writes():
    import pytest

    validate_code("import pandas as pd\npd.read_parquet('/telemetry/metric_stats.parquet')")
    with pytest.raises(PermissionError):
        validate_code("import subprocess")
    with pytest.raises(PermissionError):
        validate_code("open('/tmp/x', 'w')")
    with pytest.raises(PermissionError):
        validate_code("df.to_csv('/tmp/x.csv')")


def test_rca_agent_replica_keeps_controller_executor_architecture(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = OpenRCADataset(root)
    prepared_root = tmp_path / "prepared"
    prepare_dataset(dataset, prepared_root, chunksize=2)
    prepared = PreparedOpenRCA(prepared_root)
    calls = []

    def responder(messages, tools):
        system = messages[0]["content"]
        calls.append(system)
        if system.startswith("You are the Controller"):
            return {
                "content": json.dumps({
                    "analysis": "Fixture telemetry is sufficient.",
                    "completed": True,
                    "instruction": "Conclude.",
                })
            }
        return {
            "content": json.dumps({
                "component": "docker_001",
                "reason": "CPU fault",
                "occurrence_time": "2020-04-11 00:15:00",
                "confidence": 0.8,
                "rationale": "fixture",
            })
        }

    class FakeSandbox:
        def __init__(self, _prepared, _row_id):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _code):
            return {"ok": True, "output": "fixture"}

    result = run_rca_agent_replica(
        prepared,
        dataset.get_runtime_task(0),
        llm=LLMClient(responder=responder, cache_enabled=False),
        sandbox_factory=FakeSandbox,
    )

    assert result.run_result.system == "rca_agent_replica"
    assert result.run_result.winner.component == "docker_001"
    assert result.run_result.artifacts["architecture"] == "controller_executor"
    assert any("Controller" in system for system in calls)
    assert any("final OpenRCA diagnosis" in system for system in calls)


def test_openrca_preregistration_freezes_rows_and_dataset_hashes(tmp_path):
    root = _fixture_dataset(tmp_path, task_ids=("task_1", "task_4", "task_7"))
    dataset = OpenRCADataset(root)

    payload = build_preregistration(
        dataset,
        systems=["single_react", "shardrca_full"],
        row_ids=[0, 2],
        seed=9,
        model="test-model",
    )

    assert payload["status"] == "frozen"
    assert payload["row_selection"]["row_ids"] == [0, 2]
    assert payload["row_selection"]["difficulty_counts"] == {"easy": 1, "hard": 1}
    assert payload["dataset"]["query_sha256"]
    assert payload["dataset"]["telemetry_manifest_sha256"]
    assert payload["runtime_safety"]["evaluator_only_fields"] == ["scoring_points"]
    assert "scoring_points" not in payload["runtime_safety"]["runtime_task_fields"]
    assert "--row-ids 0,2" in payload["commands"][0]
    assert "--systems single_react,shardrca_full" in payload["commands"][0]
    assert "telco_mas.openrca.result_analysis" in payload["analysis_command"]
    assert "--baseline strongest_single" in payload["analysis_command"]


def test_openrca_cli_runs_explicit_preregistered_row_ids(tmp_path):
    from telco_mas.openrca.cli import main as openrca_main

    root = _fixture_dataset(tmp_path, task_ids=("task_1", "task_4", "task_7"))
    out = tmp_path / "openrca_rows.json"

    rc = openrca_main([
        "--data-dir", str(root),
        "--mode", "smoke",
        "--row-ids", "0,2",
        "--out", str(out),
    ])

    payload = json.loads(out.read_text())
    assert rc == 0
    assert [row["row_id"] for row in payload["rows"]] == [0, 2]
    assert [row["task_index"] for row in payload["rows"]] == ["task_1", "task_7"]
    assert {row["system"] for row in payload["rows"]} == {"smoke"}


def test_openrca_cli_requires_explicit_live_llm_confirmation(tmp_path, capsys):
    from telco_mas.openrca.cli import main as openrca_main

    root = _fixture_dataset(tmp_path)
    out = tmp_path / "openrca_llm.json"

    rc = openrca_main([
        "--data-dir", str(root),
        "--mode", "llm",
        "--row-ids", "0",
        "--out", str(out),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--confirm-live-llm" in captured.err
    assert not out.exists()


def test_openrca_cli_runs_frozen_prereg_as_one_paired_file(tmp_path):
    from telco_mas.openrca.cli import main as openrca_main

    root = _fixture_dataset(tmp_path, task_ids=("task_1", "task_4", "task_7"))
    dataset = OpenRCADataset(root)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(build_preregistration(
        dataset,
        systems=["smoke_a", "smoke_b"],
        row_ids=[0, 2],
        model="test-model",
    )), encoding="utf-8")
    out = tmp_path / "openrca_paired.json"

    rc = openrca_main([
        "--data-dir", str(root),
        "--mode", "smoke",
        "--prereg", str(prereg_path),
        "--out", str(out),
    ])

    payload = json.loads(out.read_text())
    analysis = analyze_openrca_results([out], baseline="smoke_a", treatment="smoke_b")
    assert rc == 0
    assert payload["meta"]["systems"] == ["smoke_a", "smoke_b"]
    assert payload["meta"]["rows"] == 2
    assert len(payload["rows"]) == 4
    assert {row["row_id"] for row in payload["rows"]} == {0, 2}
    assert {row["system"] for row in payload["rows"]} == {"smoke_a", "smoke_b"}
    assert set(payload["summary_by_system"]) == {"smoke_a", "smoke_b"}
    assert analysis["paired"]["strict_correct"]["mcnemar"]["paired_cases"] == 2


def test_openrca_result_analysis_reports_paired_effects_and_usage(tmp_path):
    baseline = tmp_path / "baseline.json"
    treatment = tmp_path / "treatment.json"
    baseline.write_text(json.dumps({
        "meta": {"system": "single_react_sc"},
        "rows": [
            _result_row(0, "task_1", score=1.0, strict=True, tokens=10),
            _result_row(1, "task_7", score=0.0, strict=False, tokens=20),
        ],
    }), encoding="utf-8")
    treatment.write_text(json.dumps({
        "meta": {"system": "shardrca_full"},
        "rows": [
            _result_row(0, "task_1", score=1.0, strict=True, tokens=30),
            _result_row(1, "task_7", score=1.0, strict=True, tokens=40),
        ],
    }), encoding="utf-8")

    analysis = analyze_openrca_results([baseline, treatment])

    assert analysis["benchmark"] == "openrca_telecom"
    assert analysis["summary"]["single_react_sc"]["strict_correct"] == 0.5
    assert analysis["summary"]["shardrca_full"]["strict_correct"] == 1.0
    assert analysis["paired"]["strict_correct"]["mcnemar"]["treatment_only_correct"] == 1
    assert analysis["paired"]["score"]["effect"]["mean_difference"] == 0.5
    assert analysis["clear_win_gate"]["absolute_delta"] == 0.5
    assert analysis["clear_win_gate"]["passed"] is False
    assert analysis["disagreement_count"] == 1
    assert analysis["usage"]["shardrca_full"]["total_tokens"] == 70


def test_openrca_result_analysis_resolves_strongest_single_baseline(tmp_path):
    weak = tmp_path / "weak.json"
    strong = tmp_path / "strong.json"
    treatment = tmp_path / "treatment.json"
    weak.write_text(json.dumps({
        "meta": {"system": "single_react_sc"},
        "rows": [
            _result_row(0, "task_1", score=0.0, strict=False, tokens=10),
            _result_row(1, "task_7", score=0.0, strict=False, tokens=10),
        ],
    }), encoding="utf-8")
    strong.write_text(json.dumps({
        "meta": {"system": "single_equal_tokens"},
        "rows": [
            _result_row(0, "task_1", score=1.0, strict=True, tokens=100),
            _result_row(1, "task_7", score=1.0, strict=True, tokens=100),
        ],
    }), encoding="utf-8")
    treatment.write_text(json.dumps({
        "meta": {"system": "shardrca_full"},
        "rows": [
            _result_row(0, "task_1", score=1.0, strict=True, tokens=30),
            _result_row(1, "task_7", score=1.0, strict=True, tokens=40),
        ],
    }), encoding="utf-8")

    analysis = analyze_openrca_results([weak, strong, treatment], baseline="strongest_single")

    assert analysis["baseline"] == "single_equal_tokens"
    assert analysis["baseline_selection"]["method"].startswith("strongest_single")
    assert analysis["clear_win_gate"]["absolute_delta"] == 0.0
    assert analysis["clear_win_gate"]["passed"] is False


def test_openrca_protocol_gate_requires_rca_agent_and_reports_strong_mechanism(tmp_path):
    result = tmp_path / "protocol.json"
    rows = []
    for row_id in range(8):
        for system, strict in (
            ("single_react_sc", False),
            ("rca_agent_replica", False),
            ("same_board_single", False),
            ("shardrca_full", True),
        ):
            row = _result_row(
                row_id,
                "task_1",
                score=1.0 if strict else 0.0,
                strict=strict,
                tokens=10,
            )
            row["system"] = system
            row["volume_bin"] = "high"
            rows.append(row)
    result.write_text(json.dumps({"meta": {"systems": []}, "rows": rows}), encoding="utf-8")

    analysis = analyze_openrca_results([result])

    assert analysis["clear_win_gate"]["protocol_complete"] is True
    assert analysis["clear_win_gate"]["passed"] is True
    assert analysis["clear_win_gate"]["strong_mechanism_passed"] is True
    assert analysis["comparisons"]["architecture_baseline"]["baseline"] == "rca_agent_replica"
    assert analysis["volume_analysis"]["positive_against_confirmatory"] is True


def _fixture_dataset(root: Path, *, task_ids=("task_7",)) -> Path:
    telecom = root / "Telecom"
    metric_dir = telecom / "telemetry" / "2020_04_11" / "metric"
    trace_dir = telecom / "telemetry" / "2020_04_11" / "trace"
    metric_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace_span.csv").write_text(
        "callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,dsName,serviceName\n",
        encoding="utf-8",
    )
    (metric_dir / "metric_container.csv").write_text(
        "itemid,name,bomc_id,timestamp,value,cmdb_id\n",
        encoding="utf-8",
    )
    rows = ["task_index,instruction,scoring_points"]
    for task_id in task_ids:
        rows.append(
            f'"{task_id}","Find the root cause between 2020-04-11 00:00:00 and 2020-04-11 00:30:00","The only predicted root cause component is docker_001"'
        )
    (telecom / "query.csv").write_text("\n".join(rows), encoding="utf-8")
    return root


def _result_row(row_id: int, task_index: str, *, score: float, strict: bool, tokens: int) -> dict:
    return {
        "row_id": row_id,
        "task_index": task_index,
        "instruction": "Find the root cause.",
        "prediction": "{}",
        "scoring_points": "hidden in analysis artifact",
        "passed": "",
        "failed": "",
        "score": score,
        "strict_correct": strict,
        "latency_s": 1.0,
        "total_tokens": tokens,
        "llm_calls": 1,
        "tool_calls": 2,
    }


def _prediction(component: str, reason: str, occurrence: str) -> str:
    return json.dumps({
        "1": {
            "root cause component": component,
            "root cause reason": reason,
            "root cause occurrence datetime": occurrence,
        }
    })


def _ms(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(dt.timestamp() * 1000)

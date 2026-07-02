import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telco_mas.openrca.dataset import OpenRCADataset
from telco_mas.openrca.evaluator import evaluate_prediction
from telco_mas.openrca.formatter import format_prediction
from telco_mas.openrca.schemas import OpenRCAPredictionItem, OpenRCAPredictionOutput
from telco_mas.openrca.tools import OpenRCATelemetryTools


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


def _fixture_dataset(root: Path) -> Path:
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
    (telecom / "query.csv").write_text(
        "\n".join([
            "task_index,instruction,scoring_points",
            '"task_7","Find the root cause between 2020-04-11 00:00:00 and 2020-04-11 00:30:00","The only predicted root cause component is docker_001"',
        ]),
        encoding="utf-8",
    )
    return root


def _ms(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(dt.timestamp() * 1000)


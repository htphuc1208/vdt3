import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telco_mas.evaluation.benchmark_readiness import build_readiness_report
from telco_mas.openrca.dataset import OpenRCADataset
from telco_mas.openrca.prereg import build_preregistration as build_openrca_prereg


def test_readiness_reports_missing_data(tmp_path):
    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
    )

    assert report["headline_ready"] is False
    assert report["fallback_ready"] is False
    assert report["synthetic_ready"] is False
    tn_rca = report["benchmarks"]["tn_rca530"]
    assert tn_rca["status"] == "source_only_no_artifact"
    assert tn_rca["artifact_status"] == "no_official_download_url_configured"
    assert tn_rca["ready_for_headline"] is False
    assert tn_rca["benchmark_claims"]["scenario_count"] == 530
    assert "https://arxiv.org/abs/2507.18190" in tn_rca["sources"]
    openrca = report["benchmarks"]["openrca_telecom"]
    assert openrca["status"] == "missing_data"
    assert openrca["source"] == "https://github.com/microsoft/OpenRCA"
    assert openrca["google_drive_file_id"] == "1cyOKpqyAP4fy-QiJ6a_cKuwR7D46zyVe"
    assert openrca["download_command"] == "bash scripts/download_openrca_telecom.sh --extract"


def test_readiness_next_action_surfaces_latest_openrca_download_blocker(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "openrca_telecom_download_attempt_2026-07-04.json").write_text(
        json.dumps({
            "status": "blocked_google_drive_quota",
            "reason": "quota",
        }),
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        report = build_readiness_report(
            openrca_data_dir=tmp_path / "missing_openrca",
            openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        )
    finally:
        os.chdir(cwd)

    assert "OpenRCA Google Drive quota" in report["next_action"]
    assert "TN-RCA530 artifact" in report["next_action"]


def test_readiness_accepts_matching_openrca_prereg(tmp_path):
    openrca_root = _openrca_fixture(tmp_path / "openrca")
    prereg_path = tmp_path / "openrca_prereg.json"
    dataset = OpenRCADataset(openrca_root)
    prereg_path.write_text(json.dumps(build_openrca_prereg(dataset, row_ids=[0, 1])), encoding="utf-8")

    report = build_readiness_report(openrca_data_dir=openrca_root, openrca_prereg=prereg_path)

    assert report["headline_ready"] is True
    assert report["benchmarks"]["openrca_telecom"]["status"] == "ready"
    assert report["benchmarks"]["openrca_telecom"]["prereg"]["matches"] is True


def test_readiness_rejects_openrca_prereg_with_no_confirmatory_rows(tmp_path):
    openrca_root = _openrca_fixture(tmp_path / "openrca")
    prereg_path = tmp_path / "openrca_prereg.json"
    dataset = OpenRCADataset(openrca_root)
    prereg_path.write_text(
        json.dumps(build_openrca_prereg(dataset, row_ids=[0, 1], contaminated_row_ids=[0, 1])),
        encoding="utf-8",
    )

    report = build_readiness_report(openrca_data_dir=openrca_root, openrca_prereg=prereg_path)

    status = report["benchmarks"]["openrca_telecom"]
    assert report["headline_ready"] is False
    assert status["status"] == "needs_frozen_prereg"
    assert status["prereg"]["matches"] is False
    assert status["prereg"]["confirmatory_row_count"] == 0


def test_readiness_rejects_stale_openrca_prereg(tmp_path):
    openrca_root = _openrca_fixture(tmp_path / "openrca")
    prereg_path = tmp_path / "openrca_prereg.json"
    dataset = OpenRCADataset(openrca_root)
    prereg_path.write_text(json.dumps(build_openrca_prereg(dataset, row_ids=[0])), encoding="utf-8")
    query = openrca_root / "Telecom" / "query.csv"
    query.write_text(query.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = build_readiness_report(openrca_data_dir=openrca_root, openrca_prereg=prereg_path)

    status = report["benchmarks"]["openrca_telecom"]
    assert report["headline_ready"] is False
    assert status["status"] == "needs_frozen_prereg"
    assert status["prereg"]["query_sha256_matches"] is False


def _openrca_fixture(root: Path) -> Path:
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
        "\n".join([
            "itemid,name,bomc_id,timestamp,value,cmdb_id",
            f"1,cpu_usage,ZJ,{_ms('2020-04-11 00:00:00')},1.0,docker_001",
        ]),
        encoding="utf-8",
    )
    (telecom / "query.csv").write_text(
        "\n".join([
            "task_index,instruction,scoring_points",
            '"task_1","Find the root cause between 2020-04-11 00:00:00 and 2020-04-11 00:30:00","The only predicted root cause component is docker_001"',
            '"task_7","Find the root cause between 2020-04-11 00:00:00 and 2020-04-11 00:30:00","The only predicted root cause component is docker_002"',
        ]),
        encoding="utf-8",
    )
    return root


def _ms(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(dt.timestamp() * 1000)

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from telco_mas.evaluation.benchmark_readiness import build_readiness_report
from telco_mas.openrca.dataset import OpenRCADataset
from telco_mas.openrca.prereg import build_preregistration as build_openrca_prereg
from telco_mas.synthetic_telco.dataset import build_dataset
from telco_mas.synthetic_telco.prereg import build_preregistration as build_synthetic_prereg
from telco_mas.telelogs.dataset import TeleLogsDataset
from telco_mas.telelogs.prereg import build_preregistration as build_official_telelogs_prereg
from telco_mas.telelogs_agent.dataset import TeleLogsAgentDataset
from telco_mas.telelogs_agent.prereg import build_preregistration as build_telelogs_prereg


def test_readiness_reports_missing_data(tmp_path):
    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
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
    telelogs_agent = report["benchmarks"]["telelogs_agent"]
    assert telelogs_agent["status"] == "missing_data"
    assert telelogs_agent["source"] == "https://huggingface.co/datasets/netop/TeleLogsAgent"
    assert telelogs_agent["repo_id"] == "netop/TeleLogsAgent"
    assert telelogs_agent["download_command"] == "scripts/download_telelogs_agent.sh"
    assert telelogs_agent["runner_modes"]["tool"].startswith("official")
    assert report["benchmarks"]["telelogs"]["status"] == "missing_data"
    assert report["benchmarks"]["telelogs"]["ready_for_fallback"] is False
    assert report["benchmarks"]["telco_troubleshooting_challenge"]["status"] == "missing_data"


def test_readiness_next_action_surfaces_latest_access_blockers(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "telelogs_agent_download_check_2026-07-04.json").write_text(
        json.dumps({
            "status": "blocked_token_not_authorized",
            "reason": "not authorized",
        }),
        encoding="utf-8",
    )
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
            telelogs_data_dir=tmp_path / "missing_telelogs",
            telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
            synthetic_telco_path=tmp_path / "missing_synthetic.json",
        )
    finally:
        os.chdir(cwd)

    assert "Wait for Hugging Face approval for TeleLogsAgent" in report["next_action"]
    assert "Google Drive quota" in report["next_action"]
    assert "TN-RCA530 artifact" in report["next_action"]


def test_readiness_accepts_matching_openrca_prereg(tmp_path):
    openrca_root = _openrca_fixture(tmp_path / "openrca")
    prereg_path = tmp_path / "openrca_prereg.json"
    dataset = OpenRCADataset(openrca_root)
    prereg_path.write_text(json.dumps(build_openrca_prereg(dataset, row_ids=[0, 1])), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=openrca_root,
        openrca_prereg=prereg_path,
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
    )

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

    report = build_readiness_report(
        openrca_data_dir=openrca_root,
        openrca_prereg=prereg_path,
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
    )

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

    report = build_readiness_report(
        openrca_data_dir=openrca_root,
        openrca_prereg=prereg_path,
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
    )

    status = report["benchmarks"]["openrca_telecom"]
    assert report["headline_ready"] is False
    assert status["status"] == "needs_frozen_prereg"
    assert status["prereg"]["query_sha256_matches"] is False


def test_readiness_accepts_matching_telelogs_fallback_prereg(tmp_path):
    telelogs_root = _telelogs_fixture(tmp_path / "telelogs")
    prereg_path = tmp_path / "telelogs_prereg.json"
    dataset = TeleLogsAgentDataset(telelogs_root)
    prereg_path.write_text(json.dumps(build_telelogs_prereg(dataset, limit_per_set=1)), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=telelogs_root,
        telelogs_prereg=prereg_path,
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
    )

    assert report["headline_ready"] is False
    assert report["fallback_ready"] is True
    assert report["benchmarks"]["telelogs_agent"]["status"] == "ready_fallback"
    assert report["benchmarks"]["telelogs_agent"]["prereg"]["matches"] is True
    assert "tool" in report["benchmarks"]["telelogs_agent"]["runner_modes"]


def test_readiness_tracks_netop_candidates_without_marking_ready(tmp_path):
    telelogs_root = Path("data/telelogs")
    challenge_root = Path("data/telco_troubleshooting_challenge")
    local_telelogs = tmp_path / telelogs_root
    local_challenge = tmp_path / challenge_root
    (local_telelogs / "troubleshooting").mkdir(parents=True)
    (local_challenge / "track_a").mkdir(parents=True)
    (local_telelogs / "troubleshooting" / "test.json").write_text(
        json.dumps([{"question": "x", "root_causes": ["speed"]}]),
        encoding="utf-8",
    )
    (local_challenge / "track_a" / "train.csv").write_text("id,question,answer\n1,x,y\n", encoding="utf-8")

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        report = build_readiness_report(
            openrca_data_dir=tmp_path / "missing_openrca",
            openrca_prereg=tmp_path / "missing_openrca_prereg.json",
            telelogs_data_dir=tmp_path / "missing_telelogs_agent",
            telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
            synthetic_telco_path=tmp_path / "missing_synthetic.json",
        )
    finally:
        os.chdir(cwd)

    assert report["fallback_ready"] is False
    assert report["benchmarks"]["telelogs"]["status"] == "needs_frozen_prereg"
    assert report["benchmarks"]["telelogs"]["ready_for_fallback"] is False
    assert report["benchmarks"]["telco_troubleshooting_challenge"]["status"] == "needs_adapter"
    assert report["benchmarks"]["telco_troubleshooting_challenge"]["ready_for_fallback"] is False


def test_readiness_accepts_matching_official_telelogs_prereg(tmp_path):
    telelogs_root = _official_telelogs_fixture(tmp_path / "telelogs")
    prereg_path = tmp_path / "official_telelogs_prereg.json"
    dataset = TeleLogsDataset(telelogs_root)
    prereg_path.write_text(json.dumps(build_official_telelogs_prereg(
        dataset,
        split="test",
        row_ids=[0, 1],
    )), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs_agent",
        telelogs_prereg=tmp_path / "missing_telelogs_agent_prereg.json",
        official_telelogs_data_dir=telelogs_root,
        official_telelogs_prereg=prereg_path,
        synthetic_telco_path=tmp_path / "missing_synthetic.json",
    )

    assert report["headline_ready"] is False
    assert report["fallback_ready"] is True
    assert report["benchmarks"]["telelogs"]["status"] == "ready_fallback"
    assert report["benchmarks"]["telelogs"]["prereg"]["matches"] is True


def test_readiness_accepts_valid_synthetic_telco_artifact(tmp_path):
    synthetic_path = tmp_path / "synthetic_telco.json"
    prereg_path = tmp_path / "synthetic_prereg.json"
    synthetic_path.write_text(json.dumps(build_dataset(suite="telco_v3")), encoding="utf-8")
    prereg_path.write_text(json.dumps(build_synthetic_prereg(synthetic_path)), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=synthetic_path,
        synthetic_telco_prereg=prereg_path,
    )

    assert report["headline_ready"] is False
    assert report["fallback_ready"] is False
    assert report["synthetic_ready"] is True
    assert report["benchmarks"]["synthetic_telco_v3"]["status"] == "ready_synthetic_fallback"
    assert report["benchmarks"]["synthetic_telco_v3"]["prereg"]["matches"] is True


def test_readiness_rejects_synthetic_telco_without_prereg(tmp_path):
    synthetic_path = tmp_path / "synthetic_telco.json"
    synthetic_path.write_text(json.dumps(build_dataset(suite="telco_v3")), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=synthetic_path,
        synthetic_telco_prereg=tmp_path / "missing_synthetic_prereg.json",
    )

    status = report["benchmarks"]["synthetic_telco_v3"]
    assert report["synthetic_ready"] is False
    assert status["status"] == "needs_frozen_prereg"
    assert status["validation"]["ok"] is True
    assert status["prereg"]["exists"] is False


def test_readiness_rejects_stale_synthetic_telco_prereg(tmp_path):
    synthetic_path = tmp_path / "synthetic_telco.json"
    prereg_path = tmp_path / "synthetic_prereg.json"
    payload = build_dataset(suite="telco_v3")
    synthetic_path.write_text(json.dumps(payload), encoding="utf-8")
    prereg_path.write_text(json.dumps(build_synthetic_prereg(synthetic_path)), encoding="utf-8")

    payload["cases"] = payload["cases"][:-1]
    synthetic_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=synthetic_path,
        synthetic_telco_prereg=prereg_path,
    )

    status = report["benchmarks"]["synthetic_telco_v3"]
    assert report["synthetic_ready"] is False
    assert status["status"] == "needs_frozen_prereg"
    assert status["prereg"]["sha256_matches"] is False
    assert status["prereg"]["case_count_matches"] is False
    assert status["prereg"]["runtime_case_ids_match"] is False


def test_readiness_accepts_frozen_telco_v4_confirmatory_protocol(tmp_path):
    synthetic_path = tmp_path / "synthetic_telco_v4.json"
    prereg_path = tmp_path / "synthetic_v4_prereg.json"
    synthetic_path.write_text(
        json.dumps(build_dataset(
            suite="telco_v4",
            seed=20260703,
            seed_source="unit-test-fixed",
        )),
        encoding="utf-8",
    )
    prereg_path.write_text(
        json.dumps(build_synthetic_prereg(
            synthetic_path,
            systems=["full", "single"],
            model="test-model",
            runs=3,
            algorithm_id="git:0123456789abcdef",
        )),
        encoding="utf-8",
    )

    report = build_readiness_report(
        openrca_data_dir=tmp_path / "missing_openrca",
        openrca_prereg=tmp_path / "missing_openrca_prereg.json",
        telelogs_data_dir=tmp_path / "missing_telelogs",
        telelogs_prereg=tmp_path / "missing_telelogs_prereg.json",
        synthetic_telco_path=synthetic_path,
        synthetic_telco_prereg=prereg_path,
    )

    status = report["benchmarks"]["synthetic_telco"]
    assert report["synthetic_ready"] is True
    assert status["suite"] == "telco_v4"
    assert status["prereg"]["content_sha256_matches"] is True
    assert status["prereg"]["source_scenario_ids_match"] is True
    assert status["prereg"]["v4_protocol_complete"] is True


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


def _telelogs_fixture(root: Path) -> Path:
    for name in ("TS1", "TS2", "TS3"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "test.json").write_text(
            json.dumps([
                {"id": f"{name}-1", "question": "Diagnose the 5G failure.", "root_cause": "hidden"},
                {"id": f"{name}-2", "question": "Diagnose the 5G failure.", "answer": "hidden"},
            ]),
            encoding="utf-8",
        )
    return root


def _official_telelogs_fixture(root: Path) -> Path:
    folder = root / "troubleshooting"
    folder.mkdir(parents=True)
    (folder / "test.json").write_text(
        json.dumps([
            {"id": "tl-1", "question": "x", "root_causes": ["speed"]},
            {"id": "tl-2", "question": "y", "root_cause": "downtilt"},
        ]),
        encoding="utf-8",
    )
    return root


def _ms(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(dt.timestamp() * 1000)

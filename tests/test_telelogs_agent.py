import json
from pathlib import Path

from telco_mas.llm import LLMClient
from telco_mas.telelogs_agent import cli as telelogs_cli
from telco_mas.telelogs_agent.dataset import TeleLogsAgentDataset
from telco_mas.telelogs_agent.evaluator import evaluator_labels, score_prediction
from telco_mas.telelogs_agent.http_tools import TeleLogsHTTPClient
from telco_mas.telelogs_agent.prereg import build_preregistration
from telco_mas.telelogs_agent.result_analysis import analyze_results


def test_telelogs_agent_runtime_task_strips_label_like_fields(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = TeleLogsAgentDataset(root)

    task = dataset.get_runtime_task("TS1", 0)
    payload_blob = json.dumps(task["payload"])

    assert task["scenario_set"] == "TS1"
    assert task["task_id"] == "case-1"
    assert "question" in task["payload"]
    assert "root_cause" not in payload_blob
    assert "gold answer" not in payload_blob
    assert "expected" not in payload_blob


def test_telelogs_agent_preregistration_freezes_all_sets(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = TeleLogsAgentDataset(root)

    payload = build_preregistration(
        dataset,
        systems=["single_react", "shardrca_full"],
        limit_per_set=1,
        seed=3,
        model="test-model",
    )

    assert payload["status"] == "frozen"
    assert payload["benchmark"]["role_in_claim"].startswith("fallback")
    assert payload["dataset"]["counts"] == {"TS1": 2, "TS2": 2, "TS3": 2}
    assert set(payload["row_selection"]["selected"]) == {"TS1", "TS2", "TS3"}
    assert all(len(ids) == 1 for ids in payload["row_selection"]["selected"].values())
    assert payload["dataset"]["manifest_sha256"]
    assert payload["runtime_safety"]["forbidden_runtime_inputs"]


def test_telelogs_agent_evaluator_uses_only_evaluator_side_labels(tmp_path):
    root = _fixture_dataset(tmp_path)
    dataset = TeleLogsAgentDataset(root)
    raw = dataset.rows("TS1")[0]
    runtime = dataset.get_runtime_task("TS1", 0)

    assert "hidden interference" not in json.dumps(runtime)
    assert "hidden interference" in evaluator_labels(raw)
    score = score_prediction(raw, {"root_cause": "Hidden interference near the serving cell"})
    miss = score_prediction(raw, {"root_cause": "congestion"})

    assert score.score_available is True
    assert score.strict_correct is True
    assert miss.strict_correct is False


def test_telelogs_agent_profile_cli_and_analysis(tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    result_path = tmp_path / "profile.json"

    rc = telelogs_cli.main([
        "--data-dir",
        str(root),
        "--systems",
        "single_react_sc,shardrca_full",
        "--limit-per-set",
        "1",
        "--mode",
        "profile",
        "--out",
        str(result_path),
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    analysis = analyze_results([result_path], baseline="strongest_single")
    assert rc == 0
    assert payload["meta"]["status"] == "completed"
    assert len(payload["rows"]) == 6
    assert set(payload["summary"]) == {"single_react_sc", "shardrca_full"}
    assert analysis["benchmark"] == "telelogs_agent"
    assert analysis["modes"] == ["profile"]
    assert analysis["official_tool_mode"] is False
    assert analysis["evidence_mode"] == "staged_or_mixed"
    assert analysis["clear_win_gate"]["passed"] is False
    assert analysis["paired"]["strict_correct"]["mcnemar"]["paired_cases"] == 3


def test_telelogs_agent_llm_cli_with_stubbed_client(monkeypatch, tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    result_path = tmp_path / "llm.json"

    def responder(messages, tools):
        return {
            "content": json.dumps({
                "root_cause": "hidden interference",
                "solution": "adjust serving cell",
                "confidence": 0.7,
                "rationale": "stubbed label-safe reasoning",
            }),
            "tool_calls": [],
        }

    monkeypatch.setattr(
        telelogs_cli,
        "LLMClient",
        lambda *args, **kwargs: LLMClient(responder=responder, cache_enabled=False),
    )
    monkeypatch.setattr(telelogs_cli, "get_settings", lambda: type("S", (), {"has_api_key": True})())

    rc = telelogs_cli.main([
        "--data-dir",
        str(root),
        "--systems",
        "single_react",
        "--limit-per-set",
        "1",
        "--mode",
        "llm",
        "--out",
        str(result_path),
        "--no-cache",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["strict_correct"] is True
    assert payload["rows"][0]["llm_calls"] == 1


def test_telelogs_agent_http_tool_client_maps_official_endpoints():
    calls = []

    def transport(method, path, params, headers, timeout):
        calls.append({
            "method": method,
            "path": path,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        })
        if path == "/tools":
            return {
                "tools": [
                    {"name": "/scenario", "description": "scenario context"},
                    {"name": "serving-cell-rsrp", "parameters": {"type": "object", "properties": {}}},
                ]
            }
        if path == "/serving-cell-rsrp":
            return {"rsrp": [-95, -111], "scenario": headers["X-Scenario-Id"]}
        return {"ok": True}

    client = TeleLogsHTTPClient(
        "http://telelogs.test",
        scenario_id="case-1",
        timeout=3.0,
        transport=transport,
    )

    specs = client.tools_spec()
    result = client.dispatch("serving_cell_rsrp", {"ue_id": "UE-1"})

    names = [spec["function"]["name"] for spec in specs]
    assert names == ["scenario", "serving_cell_rsrp"]
    assert json.loads(result)["scenario"] == "case-1"
    assert calls[-1]["path"] == "/serving-cell-rsrp"
    assert calls[-1]["params"] == {"ue_id": "UE-1"}
    assert calls[-1]["headers"]["X-Scenario-Id"] == "case-1"


def test_telelogs_agent_tool_cli_with_stubbed_client(monkeypatch, tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    result_path = tmp_path / "tool.json"
    seen_servers = []

    def responder(messages, tools):
        if tools and not any(message.get("role") == "tool" for message in messages):
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "serving_cell_rsrp", "arguments": "{}"},
                    }
                ],
            }
        return {
            "content": json.dumps({
                "root_cause": "hidden interference",
                "solution": "adjust serving cell",
                "confidence": 0.8,
                "evidence": ["serving_cell_rsrp showed degradation"],
                "rationale": "stubbed tool-backed reasoning",
            }),
            "tool_calls": [],
        }

    class FakeHTTPClient:
        def __init__(self, server_url, *, scenario_id, timeout=20.0, max_result_chars=16000):
            seen_servers.append((server_url, scenario_id))
            self.scenario_id = scenario_id

        def tools_spec(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "serving_cell_rsrp",
                        "description": "RSRP",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def dispatch(self, name, arguments):
            assert name == "serving_cell_rsrp"
            return json.dumps({"scenario_id": self.scenario_id, "rsrp": [-95, -112]})

    monkeypatch.setattr(
        telelogs_cli,
        "LLMClient",
        lambda *args, **kwargs: LLMClient(responder=responder, cache_enabled=False),
    )
    monkeypatch.setattr(telelogs_cli, "TeleLogsHTTPClient", FakeHTTPClient)
    monkeypatch.setattr(telelogs_cli, "get_settings", lambda: type("S", (), {"has_api_key": True})())

    rc = telelogs_cli.main([
        "--data-dir",
        str(root),
        "--systems",
        "single_react",
        "--limit-per-set",
        "1",
        "--mode",
        "tool",
        "--out",
        str(result_path),
        "--server-url-map",
        "TS1=http://ts1.test,TS2=http://ts2.test,TS3=http://ts3.test",
        "--no-cache",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["meta"]["mode"] == "tool"
    assert len(payload["rows"]) == 3
    assert payload["rows"][0]["strict_correct"] is True
    assert payload["rows"][0]["tool_calls"] == 1
    assert payload["rows"][0]["tool_failures"] == 0
    assert payload["rows"][0]["tool_failure_rate"] == 0.0
    assert payload["rows"][0]["tool_call_efficiency"] == 1.0
    assert payload["meta"]["server_url_by_set"] == {
        "TS1": "http://ts1.test",
        "TS2": "http://ts2.test",
        "TS3": "http://ts3.test",
    }
    assert ("http://ts1.test", "case-1") in seen_servers
    assert ("http://ts2.test", "case-1") in seen_servers
    assert ("http://ts3.test", "case-1") in seen_servers
    assert "hidden interference" not in json.dumps(TeleLogsAgentDataset(root).get_runtime_task("TS1", 0))


def test_telelogs_agent_tool_cli_counts_tool_failures(monkeypatch, tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    result_path = tmp_path / "tool_failures.json"

    def responder(messages, tools):
        if tools and not any(message.get("role") == "tool" for message in messages):
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "serving_cell_rsrp", "arguments": "{}"},
                    }
                ],
            }
        return {
            "content": json.dumps({
                "root_cause": "hidden interference",
                "solution": "adjust serving cell",
                "confidence": 0.8,
                "rationale": "stubbed answer after failed tool call",
            }),
            "tool_calls": [],
        }

    class FailingHTTPClient:
        def __init__(self, server_url, *, scenario_id, timeout=20.0, max_result_chars=16000):
            self.scenario_id = scenario_id

        def tools_spec(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "serving_cell_rsrp",
                        "description": "RSRP",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def dispatch(self, name, arguments):
            raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr(
        telelogs_cli,
        "LLMClient",
        lambda *args, **kwargs: LLMClient(responder=responder, cache_enabled=False),
    )
    monkeypatch.setattr(telelogs_cli, "TeleLogsHTTPClient", FailingHTTPClient)
    monkeypatch.setattr(telelogs_cli, "get_settings", lambda: type("S", (), {"has_api_key": True})())

    rc = telelogs_cli.main([
        "--data-dir",
        str(root),
        "--systems",
        "single_react",
        "--limit-per-set",
        "1",
        "--mode",
        "tool",
        "--out",
        str(result_path),
        "--server-url",
        "http://telelogs.test",
        "--no-cache",
    ])

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["rows"][0]["strict_correct"] is True
    assert payload["rows"][0]["tool_calls"] == 1
    assert payload["rows"][0]["tool_failures"] == 1
    assert payload["rows"][0]["tool_failure_rate"] == 1.0
    assert payload["summary"]["single_react"]["tool_failure_rate"] == 1.0


def test_telelogs_agent_cli_rejects_stale_prereg(tmp_path):
    root = _fixture_dataset(tmp_path / "telelogs")
    dataset = TeleLogsAgentDataset(root)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(build_preregistration(dataset, limit_per_set=1)), encoding="utf-8")
    (root / "TS1" / "test.json").write_text(json.dumps([{"id": "changed", "question": "x"}]), encoding="utf-8")
    out = tmp_path / "out.json"

    rc = telelogs_cli.main([
        "--data-dir",
        str(root),
        "--prereg",
        str(prereg_path),
        "--out",
        str(out),
    ])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 2
    assert payload["meta"]["status"] == "skipped"
    assert "manifest differs" in payload["meta"]["reason"]


def _fixture_dataset(root: Path) -> Path:
    for name in ("TS1", "TS2", "TS3"):
        folder = root / name
        folder.mkdir(parents=True)
        rows = [
            {
                "id": "case-1",
                "question": f"Diagnose {name} throughput drop.",
                "tools": [{"name": "query_kpi"}],
                "root_cause": "hidden interference",
                "expected_answer": "gold answer",
            },
            {
                "id": "case-2",
                "question": f"Diagnose {name} handover failure.",
                "metadata": {"label": "hidden label", "visible": "cell edge symptoms"},
            },
        ]
        (folder / "test.json").write_text(json.dumps(rows), encoding="utf-8")
    return root

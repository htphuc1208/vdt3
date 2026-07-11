#!/usr/bin/env python3
"""Live browser demo for ShardRCA on a concrete OpenRCA Telecom incident.

The app intentionally uses only the Python standard library for serving the UI.
It streams a real ShardRCA run with Server-Sent Events; the page is not a static
export. Runtime inputs are label-safe, and the evaluator panel is revealed only
after the prediction is produced.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telco_mas.openrca.dataset import OpenRCADataset
from telco_mas.openrca.evaluator import build_eval_result
from telco_mas.openrca.formatter import format_prediction
from telco_mas.openrca.prepared import PreparedOpenRCA
from telco_mas.openrca.tools import candidate_catalog_for_row
from telco_mas.config import get_settings
from telco_mas.llm import LLMClient, LLMError, extract_json
from telco_mas.schemas import UsageStats
from telco_mas.shardrca.runner import run_openrca_task


DEFAULT_ROW_ID = 0
DATA_DIR = ROOT / "data" / "openrca"
PREPARED_DIR = ROOT / "data" / "openrca_prepared" / "Telecom"
DEMO_CASE_NOTES = {
    0: {
        "label": "protocol strict pass",
        "expected_score": "1.00",
        "note": "Reported OpenRCA protocol: ShardRCA LLM agents predicted the correct root cause reason.",
    },
    1: {
        "label": "protocol strict pass",
        "expected_score": "1.00",
        "note": "Reported OpenRCA protocol: ShardRCA LLM agents predicted occurrence time and reason exactly.",
    },
    3: {
        "label": "protocol strict pass",
        "expected_score": "1.00",
        "note": "Reported OpenRCA protocol: ShardRCA LLM agents predicted the correct root cause reason.",
    },
    11: {
        "label": "offline fallback pass",
        "expected_score": "0.00 reported",
        "note": "This row passes in deterministic fallback but failed in the reported LLM-agent protocol.",
    },
    12: {
        "label": "protocol fail",
        "expected_score": "0.00",
        "note": "Hard row under the reported protocol; useful for failure analysis, not as the first success demo.",
    },
    15: {
        "label": "protocol partial",
        "expected_score": "0.33",
        "note": "Reported protocol gets one field right; useful for explaining OpenRCA field-level scoring.",
    },
    17: {
        "label": "protocol strict pass",
        "expected_score": "1.00",
        "note": "Reported OpenRCA protocol: ShardRCA LLM agents predicted the correct root cause reason.",
    },
    4: {
        "label": "protocol partial",
        "expected_score": "0.33",
        "note": "Hard case for failure analysis. Do not use as the first success demo.",
    },
}

AUDITOR_SYSTEM = """You are a label-safe telecom RCA evidence auditor for a live demo.
You receive only runtime task text, telemetry evidence, ShardRCA fusion candidates,
and the final prediction. Hidden evaluator labels are not available.
Do not invent components, reasons, or evidence pointers. Do not rewrite the final answer.
Return ONLY JSON:
{"verdict":"supported|weak|contradicted",
 "summary":"one short sentence",
 "supporting_evidence":["exact evidence pointer or signal"],
 "risk":"main uncertainty or missing evidence"}"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ShardRCA live demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = _make_handler()
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"ShardRCA live demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()
    return 0


def _make_handler():
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "ShardRCADemo/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._write_html(INDEX_HTML)
            elif parsed.path == "/api/cases":
                self._write_json(_case_list())
            elif parsed.path == "/api/run":
                params = urllib.parse.parse_qs(parsed.query)
                row_id = int(params.get("row_id", [str(DEFAULT_ROW_ID)])[0])
                system = str(params.get("system", ["shardrca_full"])[0])
                llm_mode = str(params.get("llm_mode", ["raw"])[0])
                self._stream_run(row_id=row_id, system=system, llm_mode=llm_mode)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[%s] %s" % (self.log_date_time_string(), fmt % args))

        def _write_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_json(self, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream_run(self, *, row_id: int, system: str, llm_mode: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            def send(event: str, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False, default=str)
                self.wfile.write(f"event: {event}\n".encode("utf-8"))
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                _run_and_stream(row_id=row_id, system=system, llm_mode=llm_mode, send=send)
            except LLMError as exc:
                send("error", {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "hint": "Set OPENAI_API_KEY/OPENAI_MODEL in .env or switch Reasoning to deterministic.",
                })
            except Exception as exc:  # keep browser session informative
                send("error", {"message": str(exc), "type": type(exc).__name__})
            finally:
                self.close_connection = True

    return DemoHandler


def _case_list() -> dict[str, Any]:
    dataset = OpenRCADataset(DATA_DIR, dataset="Telecom")
    rows = []
    for row_id in range(min(len(dataset.rows), 18)):
        task = dataset.get_runtime_task(row_id)
        note = DEMO_CASE_NOTES.get(row_id, {
            "label": "exploratory",
            "expected_score": "unknown",
            "note": "Exploratory row. OpenRCA exact field scoring may return 0 if component, reason, or occurrence time is off.",
        })
        rows.append({
            "row_id": row_id,
            "task_index": task["task_index"],
            "instruction": task["instruction"],
            "recommended": row_id == DEFAULT_ROW_ID,
            "demo_label": note["label"],
            "expected_demo_score": note["expected_score"],
            "demo_note": note["note"],
        })
    return {
        "default_row_id": DEFAULT_ROW_ID,
        "rows": rows,
        "llm": _llm_public_settings(),
        "note": "Runtime tasks hide scoring_points; evaluator labels are shown only after prediction.",
    }


def _llm_public_settings() -> dict[str, Any]:
    settings = get_settings()
    return {
        "has_api_key": settings.has_api_key,
        "provider": settings.provider_label,
        "base_url": settings.base_url,
        "model": settings.model,
        "temperature": settings.temperature,
        "cache_enabled": settings.cache_enabled,
        "seed": settings.seed,
    }


def _normalise_llm_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases = {
        "live": "raw",
        "llm": "raw",
        "paper": "raw",
        "paper_protocol": "raw",
        "audit": "review",
        "llm_review": "review",
        "reviewed": "review",
        "raw_llm": "raw",
        "full_llm": "raw",
        "agents": "raw",
        "deterministic": "offline",
        "none": "offline",
        "no_llm": "offline",
    }
    return aliases.get(normalized, normalized)


def _llm_for_mode(mode: str) -> LLMClient | None:
    normalized = _normalise_llm_mode(mode)
    if normalized == "offline":
        return None
    if normalized not in {"review", "raw"}:
        raise ValueError(f"unsupported llm_mode: {mode}")
    llm = LLMClient(cache_enabled=None)
    if not llm.settings.has_api_key:
        raise LLMError("Live LLM mode requires OPENAI_API_KEY.")
    return llm


def _llm_runtime_metadata(llm: LLMClient | None, mode: str) -> dict[str, Any]:
    normalized = _normalise_llm_mode(mode)
    if llm is None:
        return {
            "mode": "deterministic",
            "requested_mode": normalized,
            "provider": None,
            "model": None,
            "temperature": None,
            "cache_enabled": False,
        }
    return {
        "mode": "reported_llm_agent_protocol" if normalized == "raw" else "auxiliary_llm_audit",
        "requested_mode": normalized,
        "reported_protocol": normalized == "raw",
        "included_in_claims": normalized == "raw",
        "provider": llm.settings.provider_label,
        "base_url": llm.settings.base_url,
        "model": llm.settings.model,
        "temperature": llm.settings.temperature,
        "cache_enabled": llm.settings.cache_enabled,
        "seed": llm.settings.seed,
    }


def _core_llm_for_mode(llm: LLMClient | None, mode: str) -> LLMClient | None:
    return llm if _normalise_llm_mode(mode) == "raw" else None


def _llm_audit_prediction(
    *,
    llm: LLMClient,
    task: dict[str, Any],
    prediction_text: str,
    result: Any,
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any], UsageStats]:
    payload = {
        "runtime_task": {
            "row_id": task.get("row_id"),
            "task_index": task.get("task_index"),
            "instruction": task.get("instruction"),
        },
        "final_prediction": prediction_text,
        "winner": result.winner.compact(),
        "top_findings": [finding.compact() for finding in result.board.ranked_findings(10)],
        "fusion_candidates": list(artifacts.get("fusion_candidates") or [])[:6],
        "pre_falsifier_winner": artifacts.get("pre_falsifier_winner", {}),
        "falsifier": artifacts.get("falsifier", {}),
        "label_safety": "No scoring_points or ground-truth labels are included in this audit payload.",
    }
    response = llm.chat(
        [
            {"role": "system", "content": AUDITOR_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, default=str)},
        ],
        force_json=True,
        temperature=0.0,
    )
    data = extract_json(response.content)
    if not data:
        data = {
            "verdict": "weak",
            "summary": "LLM audit response could not be parsed as JSON.",
            "supporting_evidence": [],
            "risk": "Unparsed model response.",
        }
    data.setdefault("verdict", "weak")
    data.setdefault("summary", "")
    data.setdefault("supporting_evidence", [])
    data.setdefault("risk", "")
    data["usage"] = response.usage.model_dump()
    return data, response.usage


def _run_and_stream(*, row_id: int, system: str, llm_mode: str, send) -> None:
    dataset = OpenRCADataset(DATA_DIR, dataset="Telecom")
    prepared = PreparedOpenRCA(PREPARED_DIR)
    prepared.validate_against(dataset)
    task = dataset.get_runtime_task(row_id)
    catalog = candidate_catalog_for_row(prepared, row_id)
    llm = _llm_for_mode(llm_mode)
    llm_meta = _llm_runtime_metadata(llm, llm_mode)
    core_llm = _core_llm_for_mode(llm, llm_mode)

    send("reset", {"row_id": row_id, "system": system, "llm": llm_meta})
    send("stage", {
        "key": "case",
        "title": "1. Label-safe incident loaded",
        "status": "running",
        "payload": {
            "row_id": row_id,
            "task_index": task["task_index"],
            "instruction": task["instruction"],
            "llm": llm_meta,
            "hidden_at_runtime": ["scoring_points", "true component", "true reason"],
        },
    })
    time.sleep(0.2)
    send("stage", {
        "key": "catalog",
        "title": "2. Candidate catalog from telemetry",
        "status": "running",
        "payload": {
            "component_count": len(catalog["components"]),
            "reason_count": len(catalog["reasons"]),
            "components": list(catalog["components"])[:16],
            "reasons": list(catalog["reasons"]),
            "label_derived": bool(catalog["source"].get("label_derived")),
        },
    })
    time.sleep(0.2)
    send("stage", {
        "key": "workers",
        "title": "3. Evidence-isolated workers are running",
        "status": "running",
        "payload": {
            "llm": llm_meta,
            "workers": [
                "node_metrics",
                "container_metrics",
                "service_middleware_metrics",
                "application_symptoms",
                "trace_dependencies",
            ],
            "prepared_cache": str(PREPARED_DIR.relative_to(ROOT)),
        },
    })

    started = time.time()
    prediction, result = run_openrca_task(
        dataset,
        task,
        system=system,
        llm=core_llm,
        prepared=prepared,
        finding_limit=12,
        chunksize=50_000,
    )
    prediction_text = format_prediction(prediction)
    evaluation = build_eval_result(
        row_id=row_id,
        task_index=str(task["task_index"]),
        instruction=str(task["instruction"]),
        prediction=prediction_text,
        scoring_points=dataset.get_scoring_points(row_id),
    ).to_dict()
    artifacts = result.artifacts or {}
    audit_payload: dict[str, Any] | None = None
    audit_usage = UsageStats()
    if llm is not None and _normalise_llm_mode(llm_mode) == "review":
        audit_payload, audit_usage = _llm_audit_prediction(
            llm=llm,
            task=task,
            prediction_text=prediction_text,
            result=result,
            artifacts=artifacts,
        )
    combined_usage = result.usage.add(audit_usage)
    elapsed = round(time.time() - started, 2)

    send("stage", {
        "key": "workers",
        "title": "3. Evidence-isolated workers completed",
        "status": "done",
        "payload": {
            "latency_s": result.latency_s,
            "finding_count": len(result.board.findings),
            "top_components": result.board.top_components(8),
            "worker_diagnostics": artifacts.get("worker_diagnostics", []),
        },
    })
    time.sleep(0.25)
    send("stage", {
        "key": "evidence",
        "title": "4. Blackboard evidence",
        "status": "done",
        "payload": {
            "findings": [finding.compact() for finding in result.board.ranked_findings(12)],
            "by_modality": _count_by_modality(result.board.findings),
        },
    })
    time.sleep(0.25)
    send("stage", {
        "key": "interaction",
        "title": "5. Peer interaction and posterior revision",
        "status": "done",
        "payload": {
            "diagnostics": artifacts.get("mas_interaction", {}).get("diagnostics", {}),
            "transcript": artifacts.get("mas_interaction", {}).get("transcript", [])[:16],
            "pre": artifacts.get("pre_interaction_worker_distributions", [])[:5],
            "post": artifacts.get("worker_distributions", [])[:5],
        },
    })
    time.sleep(0.25)
    send("stage", {
        "key": "fusion",
        "title": "6. Log-opinion pool fusion",
        "status": "done",
        "payload": {
            "round_1": artifacts.get("round_1", [])[:8],
            "round_2": artifacts.get("round_2", [])[:8],
            "fusion_candidates": artifacts.get("fusion_candidates", [])[:8],
            "fusion_weights": artifacts.get("fusion_weights", {}),
            "pre_falsifier_winner": artifacts.get("pre_falsifier_winner", {}),
        },
    })
    time.sleep(0.25)
    send("stage", {
        "key": "verifier",
        "title": "7. Evidence verifier, LLM audit, and final answer",
        "status": "done",
        "payload": {
            "winner": result.winner.compact(),
            "falsifier": artifacts.get("falsifier", {}),
            "llm_audit": audit_payload,
            "prediction": prediction_text,
            "notes": result.notes,
            "usage": combined_usage.model_dump(),
            "llm": llm_meta,
            "wall_time_s": elapsed,
        },
    })
    time.sleep(0.25)
    send("stage", {
        "key": "evaluation",
        "title": "8. Evaluator panel (revealed after prediction)",
        "status": "done",
        "payload": evaluation,
    })
    send("complete", {
        "strict_correct": bool(evaluation["strict_correct"]),
        "score": evaluation["score"],
        "component": result.winner.component,
        "reason": result.winner.reason,
    })


def _count_by_modality(findings: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        modality = str(getattr(finding, "modality", "unknown"))
        counts[modality] = counts.get(modality, 0) + 1
    return counts


INDEX_HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ShardRCA Live Telecom RCA Demo</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --ink: #18202f;
      --muted: #5f6a7a;
      --line: #dce2ec;
      --panel: #ffffff;
      --blue: #2356d8;
      --green: #147a4a;
      --red: #b42318;
      --amber: #9a6700;
      --violet: #7149b8;
      --shadow: 0 16px 42px rgba(24, 32, 47, 0.09);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    header {
      padding: 24px 28px 18px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    .topline {
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
      max-width: 1440px;
      margin: 0 auto;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .controls {
      display: flex;
      align-items: end;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    select, button {
      height: 38px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 0 12px;
      font: inherit;
    }
    button {
      border: 0;
      background: var(--blue);
      color: white;
      font-weight: 800;
      cursor: pointer;
      min-width: 118px;
    }
    button:disabled { opacity: .55; cursor: wait; }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 28px 32px;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .case-card { padding: 18px; }
    .case-card h2, .work h2 {
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .case-text {
      font-size: 14px;
      color: #273142;
      border-left: 3px solid var(--blue);
      padding-left: 12px;
      margin: 12px 0 14px;
    }
    .pill-row {
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .pill {
      padding: 5px 8px;
      border-radius: 999px;
      background: #eef2f8;
      color: #334155;
      font-size: 12px;
      font-weight: 700;
    }
    .pill.good { background: #e7f6ee; color: var(--green); }
    .pill.warn { background: #fff4d6; color: var(--amber); }
    .timeline {
      padding: 6px 0 0;
      display: grid;
      gap: 8px;
    }
    .step {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      display: grid;
      gap: 4px;
      background: #fff;
    }
    .step strong { font-size: 13px; }
    .step span { color: var(--muted); font-size: 12px; }
    .step.running { border-color: #9fb8ff; background: #f3f6ff; }
    .step.done { border-color: #bde5cf; background: #f1fbf5; }
    .work {
      min-width: 0;
      padding: 16px;
    }
    .hero {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 78px;
    }
    .metric small { color: var(--muted); display: block; margin-bottom: 5px; }
    .metric b { font-size: 18px; }
    .tabs {
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 12px;
      overflow-x: auto;
    }
    .tab {
      border: 0;
      background: transparent;
      color: var(--muted);
      min-width: auto;
      height: 34px;
      padding: 0 8px;
    }
    .tab.active { color: var(--blue); border-bottom: 3px solid var(--blue); border-radius: 0; }
    .grid2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .box {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-width: 0;
    }
    .box h3 { margin: 0 0 8px; font-size: 14px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.4;
      background: #f6f8fb;
      border-radius: 7px;
      padding: 10px;
      max-height: 360px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .answer {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 8px;
      background: #f7fbff;
      border: 1px solid #cfe0ff;
      margin-bottom: 12px;
    }
    .answer strong { color: var(--blue); }
    .hidden { display: none; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .hero, .grid2 { grid-template-columns: 1fr; }
      .topline { align-items: flex-start; flex-direction: column; }
      .controls { justify-content: flex-start; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <h1>ShardRCA Live Telecom RCA Demo</h1>
        <p class="subtitle">OpenRCA Telecom case chạy trực tiếp: evidence shards → peer review → fusion → verifier → evaluator.</p>
      </div>
      <div class="controls">
        <label>Case
          <select id="caseSelect"></select>
        </label>
        <label>System
          <select id="systemSelect">
            <option value="shardrca_full">shardrca_full</option>
            <option value="no_interaction">no_interaction</option>
            <option value="no_falsifier">no_falsifier</option>
            <option value="no_refinement">no_refinement</option>
          </select>
        </label>
        <label>Reasoning
          <select id="llmModeSelect">
            <option value="raw">Reported protocol</option>
            <option value="offline">Deterministic fallback</option>
            <option value="review">Auxiliary audit</option>
          </select>
        </label>
        <button id="runBtn">Run Live</button>
      </div>
    </div>
  </header>
  <main>
    <aside class="panel case-card">
      <h2>Incident</h2>
      <div id="caseText" class="case-text">Loading cases...</div>
      <div class="pill-row">
        <span id="caseBadge" class="pill good">protocol strict pass</span>
        <span id="modeBadge" class="pill">Reported protocol</span>
        <span class="pill warn">OpenRCA exact field scoring</span>
      </div>
      <p id="caseNote" style="color:var(--muted);font-size:13px;margin:12px 0 0"></p>
      <h2 style="margin-top:18px">Run Timeline</h2>
      <div id="timeline" class="timeline"></div>
    </aside>
    <section class="panel work">
      <div id="answer" class="answer">
        <strong>Final Answer</strong>
        <span>Chưa chạy. Chọn case rồi bấm Run Live.</span>
      </div>
      <div class="hero">
        <div class="metric"><small>Component</small><b id="mComponent">-</b></div>
        <div class="metric"><small>Reason</small><b id="mReason">-</b></div>
        <div class="metric"><small>Score</small><b id="mScore">-</b></div>
        <div class="metric"><small>Latency</small><b id="mLatency">-</b></div>
        <div class="metric"><small>LLM Calls</small><b id="mLlmCalls">-</b></div>
      </div>
      <div class="tabs">
        <button class="tab active" data-tab="evidence">Evidence</button>
        <button class="tab" data-tab="agents">Agents</button>
        <button class="tab" data-tab="peer">Peer Review</button>
        <button class="tab" data-tab="fusion">Fusion</button>
        <button class="tab" data-tab="audit">LLM Audit</button>
        <button class="tab" data-tab="eval">Evaluator</button>
      </div>
      <div id="tab-evidence" class="tabpane"></div>
      <div id="tab-agents" class="tabpane hidden"></div>
      <div id="tab-peer" class="tabpane hidden"></div>
      <div id="tab-fusion" class="tabpane hidden"></div>
      <div id="tab-audit" class="tabpane hidden"></div>
      <div id="tab-eval" class="tabpane hidden"></div>
    </section>
  </main>
  <script>
    const state = { cases: [], stages: {}, source: null, llm: null };
    const $ = (id) => document.getElementById(id);
    const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const pretty = (v) => JSON.stringify(v ?? {}, null, 2);

    function setPane(name, html) { $('tab-' + name).innerHTML = html; }
    function table(rows, cols) {
      if (!rows || !rows.length) return '<pre>No data yet.</pre>';
      return `<table><thead><tr>${cols.map(c => `<th>${esc(c[0])}</th>`).join('')}</tr></thead><tbody>` +
        rows.map(r => `<tr>${cols.map(c => `<td>${esc(c[1](r))}</td>`).join('')}</tr>`).join('') +
        '</tbody></table>';
    }
    function renderTimeline() {
      const order = ['case','catalog','workers','evidence','interaction','fusion','verifier','evaluation'];
      $('timeline').innerHTML = order.map(key => {
        const s = state.stages[key] || {title: key, status: ''};
        return `<div class="step ${esc(s.status)}"><strong>${esc(s.title || key)}</strong><span>${esc(statusText(s))}</span></div>`;
      }).join('');
    }
    function statusText(s) {
      if (!s.payload) return 'waiting';
      if (s.status === 'running') return 'running...';
      if (s.key === 'workers') return `${s.payload.finding_count || 0} findings`;
      if (s.key === 'evaluation') return s.payload.strict_correct ? 'strict correct' : `partial score ${s.payload.score}`;
      return 'done';
    }
    function modeBadgeText(llm) {
      if (!llm || llm.mode === 'deterministic') return 'Deterministic fallback';
      if (llm.mode === 'reported_llm_agent_protocol') return `Reported protocol · ${llm.provider || 'LLM'} · ${llm.model || 'model'}`;
      if (llm.mode === 'auxiliary_llm_audit') return `Auxiliary audit · ${llm.provider || 'LLM'} · ${llm.model || 'model'}`;
      return `${llm.provider || 'LLM'} · ${llm.model || 'model'}`;
    }
    function updateStage(stage) {
      state.stages[stage.key] = stage;
      renderTimeline();
      const p = stage.payload || {};
      if (stage.key === 'case') {
        $('caseText').textContent = p.instruction;
        const llm = p.llm || {};
        $('modeBadge').textContent = modeBadgeText(llm);
        setPane('evidence', `<div class="grid2"><div class="box"><h3>Runtime Task</h3><pre>${esc(pretty(p))}</pre></div><div class="box"><h3>Runtime Backend</h3><pre>${esc(pretty(llm))}</pre></div></div>`);
      }
      if (stage.key === 'catalog') {
        setPane('agents', `<div class="grid2"><div class="box"><h3>Candidate Components</h3><pre>${esc((p.components || []).join('\\n'))}</pre></div><div class="box"><h3>Reasons</h3><pre>${esc((p.reasons || []).join('\\n'))}</pre></div></div>`);
      }
      if (stage.key === 'evidence') {
        setPane('evidence', `<div class="grid2"><div class="box"><h3>Top Findings</h3>${table(p.findings, [['component', r=>r.component], ['modality', r=>r.modality], ['signal', r=>r.signal], ['dir', r=>r.direction], ['score', r=>r.score]])}</div><div class="box"><h3>By Modality</h3><pre>${esc(pretty(p.by_modality))}</pre></div></div>`);
      }
      if (stage.key === 'interaction') {
        const transcript = (p.transcript || []).slice(0, 14);
        setPane('peer', `<div class="grid2"><div class="box"><h3>Diagnostics</h3><pre>${esc(pretty(p.diagnostics))}</pre></div><div class="box"><h3>Transcript</h3><pre>${esc(pretty(transcript))}</pre></div></div>`);
        setPane('agents', `<div class="grid2"><div class="box"><h3>Pre Interaction</h3><pre>${esc(pretty(p.pre))}</pre></div><div class="box"><h3>Post Interaction</h3><pre>${esc(pretty(p.post))}</pre></div></div>`);
      }
      if (stage.key === 'fusion') {
        setPane('fusion', `<div class="grid2"><div class="box"><h3>Fusion Candidates</h3>${table(p.fusion_candidates, [['component', r=>r.component], ['reason', r=>r.reason], ['confidence', r=>r.confidence], ['score', r=>r.score]])}</div><div class="box"><h3>Weights / Pre-verifier Winner</h3><pre>${esc(pretty({weights:p.fusion_weights, pre_falsifier_winner:p.pre_falsifier_winner}))}</pre></div></div>`);
      }
      if (stage.key === 'verifier') {
        $('mComponent').textContent = p.winner?.component || '-';
        $('mReason').textContent = p.winner?.reason || '-';
        $('mLatency').textContent = `${p.wall_time_s || 0}s`;
        $('mLlmCalls').textContent = String(p.usage?.llm_calls ?? 0);
        $('answer').innerHTML = `<strong>Final Answer</strong><span>${esc(p.prediction)}</span>`;
        setPane('fusion', $('tab-fusion').innerHTML + `<div class="box" style="margin-top:12px"><h3>Verifier</h3><pre>${esc(pretty(p.falsifier))}</pre></div>`);
        setPane('audit', `<div class="grid2"><div class="box"><h3>LLM Audit</h3><pre>${esc(pretty(p.llm_audit || {status:'not requested'}))}</pre></div><div class="box"><h3>Usage / Backend</h3><pre>${esc(pretty({usage:p.usage, backend:p.llm}))}</pre></div></div>`);
      }
      if (stage.key === 'evaluation') {
        $('mScore').textContent = p.strict_correct ? '1.00 ✓' : String(p.score);
        $('mScore').style.color = p.strict_correct ? 'var(--green)' : 'var(--red)';
        setPane('eval', `<div class="grid2"><div class="box"><h3>Prediction</h3><pre>${esc(p.prediction)}</pre></div><div class="box"><h3>Evaluator Result</h3><pre>${esc(pretty({score:p.score, strict_correct:p.strict_correct, passed:p.passed, failed:p.failed, scoring_points:p.scoring_points}))}</pre></div></div>`);
      }
    }
    async function loadCases() {
      const res = await fetch('/api/cases');
      const payload = await res.json();
      state.cases = payload.rows;
      state.llm = payload.llm || {};
      const rawOption = $('llmModeSelect').querySelector('option[value="raw"]');
      const reviewOption = $('llmModeSelect').querySelector('option[value="review"]');
      if (!state.llm.has_api_key) {
        rawOption.disabled = true;
        reviewOption.disabled = true;
        $('llmModeSelect').value = 'offline';
      }
      $('modeBadge').textContent = state.llm.has_api_key ? `Reported protocol · ${state.llm.provider} · ${state.llm.model}` : 'Deterministic fallback';
      $('caseSelect').innerHTML = state.cases.map(c => {
        const expected = c.expected_demo_score && c.expected_demo_score !== 'unknown' ? ` · expected ${esc(c.expected_demo_score)}` : '';
        return `<option value="${c.row_id}" ${c.recommended ? 'selected' : ''}>row ${c.row_id} · ${esc(c.demo_label || 'exploratory')}${expected}</option>`;
      }).join('');
      const selected = state.cases.find(c => c.recommended) || state.cases[0];
      if (selected) renderCase(selected);
    }
    function renderCase(c) {
      $('caseText').textContent = c.instruction;
      $('caseBadge').textContent = c.demo_label || 'exploratory';
      $('caseBadge').className = c.recommended ? 'pill good' : (String(c.expected_demo_score) === '0.00' ? 'pill warn' : 'pill');
      $('caseNote').textContent = c.demo_note || '';
    }
    function resetUI() {
      state.stages = {};
      renderTimeline();
      $('mComponent').textContent = '-';
      $('mReason').textContent = '-';
      $('mScore').textContent = '-';
      $('mScore').style.color = '';
      $('mLatency').textContent = '-';
      $('mLlmCalls').textContent = '-';
      $('answer').innerHTML = '<strong>Final Answer</strong><span>Running...</span>';
      ['evidence','agents','peer','fusion','audit','eval'].forEach(t => setPane(t, '<pre>Waiting for live events...</pre>'));
    }
    function runLive() {
      if (state.source) state.source.close();
      resetUI();
      $('runBtn').disabled = true;
      const row = $('caseSelect').value;
      const system = $('systemSelect').value;
      const llmMode = $('llmModeSelect').value;
      state.source = new EventSource(`/api/run?row_id=${encodeURIComponent(row)}&system=${encodeURIComponent(system)}&llm_mode=${encodeURIComponent(llmMode)}`);
      state.source.addEventListener('stage', ev => updateStage(JSON.parse(ev.data)));
      state.source.addEventListener('error', ev => {
        let msg = 'Stream ended.';
        let hint = '';
        try { msg = JSON.parse(ev.data).message || msg; } catch (_) {}
        try { hint = JSON.parse(ev.data).hint || ''; } catch (_) {}
        $('answer').innerHTML = `<strong>Run Error</strong><span>${esc(msg)}${hint ? ' · ' + esc(hint) : ''}</span>`;
        $('runBtn').disabled = false;
        if (state.source) state.source.close();
      });
      state.source.addEventListener('complete', ev => {
        $('runBtn').disabled = false;
        if (state.source) state.source.close();
      });
    }
    document.querySelectorAll('.tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tabpane').forEach(p => p.classList.add('hidden'));
        btn.classList.add('active');
        $('tab-' + btn.dataset.tab).classList.remove('hidden');
      });
    });
    $('caseSelect').addEventListener('change', () => {
      const c = state.cases.find(x => String(x.row_id) === $('caseSelect').value);
      if (c) renderCase(c);
    });
    $('llmModeSelect').addEventListener('change', () => {
      const value = $('llmModeSelect').value;
      if (value === 'offline') {
        $('modeBadge').textContent = 'Deterministic fallback';
      } else if (state.llm?.has_api_key) {
        $('modeBadge').textContent = value === 'raw'
          ? `Reported protocol · ${state.llm.provider} · ${state.llm.model}`
          : `Auxiliary audit · ${state.llm.provider} · ${state.llm.model}`;
      } else {
        $('modeBadge').textContent = 'Missing OPENAI_API_KEY';
      }
    });
    $('runBtn').addEventListener('click', runLive);
    renderTimeline();
    loadCases();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

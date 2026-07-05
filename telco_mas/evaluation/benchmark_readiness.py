"""Readiness checks for real/fallback RCA benchmark experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..openrca.dataset import OpenRCADataset, OpenRCADatasetError
from ..openrca.prereg import _sha256_file as _sha256_file
from ..openrca.prereg import _telemetry_manifest
from ..synthetic_telco.dataset import validate_dataset as validate_synthetic_telco
from ..synthetic_telco.prereg import _sha256_file as _sha256_synthetic_file
from ..telecomts.catalog import CATALOG_POLICY_ID, build_training_catalog, catalog_sha256
from ..telecomts.dataset import TelecomTSDataset, TelecomTSDatasetError
from ..telecomts.prereg import dataset_manifest as _telecomts_manifest
from ..telelogs.dataset import TeleLogsDataset, TeleLogsDatasetError
from ..telelogs.prereg import _manifest as _official_telelogs_manifest
from ..telelogs_agent.dataset import TeleLogsAgentDataset, TeleLogsAgentDatasetError
from ..telelogs_agent.prereg import _manifest as _telelogs_manifest


_TELELOGS_RUNNER_MODES = {
    "profile": "ingestion/scoring smoke only",
    "llm": "label-safe staged LLM prompt over stripped task JSON",
    "tool": "official TeleLogsAgent FastAPI HTTP tools via X-Scenario-Id",
}

_NETOP_DATASETS = {
    "telelogs": {
        "name": "TeleLogs",
        "repo_id": "netop/TeleLogs",
        "data_dir": "data/telelogs",
        "role": "official synthetic 5G RCA fallback; below real OpenRCA/TN-RCA evidence",
        "source": "https://huggingface.co/datasets/netop/TeleLogs",
        "expected_globs": ["troubleshooting/**/*.json", "**/*.json"],
    },
    "telco_troubleshooting_challenge": {
        "name": "Telco Troubleshooting Agentic Challenge",
        "repo_id": "netop/Telco-Troubleshooting-Agentic-Challenge",
        "data_dir": "data/telco_troubleshooting_challenge",
        "role": "official telecom agentic challenge candidate; adapter still required",
        "source": "https://huggingface.co/datasets/netop/Telco-Troubleshooting-Agentic-Challenge",
        "expected_globs": ["**/*.json", "**/*.csv"],
    },
}

_OPENRCA_TELECOM_SOURCE = {
    "source": "https://github.com/microsoft/OpenRCA",
    "google_drive_file_id": "1cyOKpqyAP4fy-QiJ6a_cKuwR7D46zyVe",
    "download_command": "bash scripts/download_openrca_telecom.sh --extract",
    "download_attempt_glob": "results/openrca_telecom_download_attempt_*.json",
}

_TELELOGS_AGENT_SOURCE = {
    "source": "https://huggingface.co/datasets/netop/TeleLogsAgent",
    "repo_id": "netop/TeleLogsAgent",
    "download_command": "scripts/download_telelogs_agent.sh",
    "download_attempt_glob": "results/telelogs_agent_download_check_*.json",
}

_TELECOMTS_SOURCE = {
    "source": "https://huggingface.co/datasets/AliMaatouk/TelecomTS",
    "paper": "https://arxiv.org/abs/2510.06063",
    "download_command": "scripts/download_telecomts.sh",
}


def build_readiness_report(
    *,
    openrca_data_dir: str | Path = "data/openrca",
    openrca_prereg: str | Path = "results/prereg_openrca_telecom_frozen.json",
    telelogs_data_dir: str | Path = "data/telelogs_agent",
    telelogs_prereg: str | Path = "results/prereg_telelogs_agent_frozen.json",
    official_telelogs_data_dir: str | Path = "data/telelogs",
    official_telelogs_prereg: str | Path = "results/prereg_telelogs_frozen.json",
    telecomts_data_dir: str | Path = "data/telecomts",
    telecomts_prereg: str | Path = "results/prereg_telecomts_frozen.json",
    synthetic_telco_path: str | Path = "results/synthetic_telco_v3_dataset.json",
    synthetic_telco_prereg: str | Path = "results/prereg_synthetic_telco_v3_frozen.json",
) -> dict[str, Any]:
    synthetic = check_synthetic_telco(synthetic_telco_path, synthetic_telco_prereg)
    benchmarks = {
        "tn_rca530": _check_tn_rca530(),
        "openrca_telecom": check_openrca(openrca_data_dir, openrca_prereg),
        "telelogs_agent": check_telelogs_agent(telelogs_data_dir, telelogs_prereg),
        "telelogs": check_official_telelogs(official_telelogs_data_dir, official_telelogs_prereg),
        "telco_troubleshooting_challenge": check_netop_dataset("telco_troubleshooting_challenge"),
        "telecomts": check_telecomts(telecomts_data_dir, telecomts_prereg),
        "synthetic_telco": synthetic,
        # Compatibility alias for existing reports/tests; inspect the returned suite field.
        "synthetic_telco_v3": synthetic,
    }
    headline_ready = any(
        item["ready_for_headline"] for item in (benchmarks["tn_rca530"], benchmarks["openrca_telecom"])
    )
    fallback_ready = (
        benchmarks["telelogs_agent"]["ready_for_fallback"]
        or benchmarks["telelogs"]["ready_for_fallback"]
    )
    synthetic_ready = (
        synthetic["ready_for_synthetic_fallback"]
        or benchmarks["telecomts"]["ready_for_synthetic_fallback"]
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "real-benchmark-first MAS RCA evaluation readiness",
        "headline_ready": headline_ready,
        "fallback_ready": fallback_ready,
        "synthetic_ready": synthetic_ready,
        "benchmarks": benchmarks,
        "next_action": _next_action(headline_ready, fallback_ready, synthetic_ready, benchmarks),
    }


def check_openrca(data_dir: str | Path, prereg_path: str | Path) -> dict[str, Any]:
    prereg_path = Path(prereg_path)
    base = {
        "source": _OPENRCA_TELECOM_SOURCE["source"],
        "google_drive_file_id": _OPENRCA_TELECOM_SOURCE["google_drive_file_id"],
        "download_command": _OPENRCA_TELECOM_SOURCE["download_command"],
        "latest_download_attempt": _latest_json(_OPENRCA_TELECOM_SOURCE["download_attempt_glob"]),
    }
    try:
        dataset = OpenRCADataset(data_dir, dataset="Telecom")
    except OpenRCADatasetError as exc:
        return {
            **base,
            "status": "missing_data",
            "ready_for_headline": False,
            "ready_for_fallback": False,
            "reason": str(exc),
            "data_dir": str(Path(data_dir)),
            "prereg_path": str(prereg_path),
        }

    current = {
        "query_sha256": _sha256_file(dataset.query_path),
        "telemetry_manifest_sha256": _telemetry_manifest(dataset.telemetry_dir)["sha256"],
        "row_count": len(dataset.rows),
    }
    prereg = _load_prereg(prereg_path)
    prereg_status = _compare_openrca_prereg(dataset, current, prereg)
    reason = None if prereg_status["matches"] else prereg_status.get("reason")
    return {
        **base,
        "status": "ready" if prereg_status["matches"] else "needs_frozen_prereg",
        "ready_for_headline": bool(prereg_status["matches"]),
        "ready_for_fallback": False,
        "reason": reason,
        "data_dir": str(dataset.root_dir),
        "query_path": str(dataset.query_path),
        "telemetry_dir": str(dataset.telemetry_dir),
        "current": current,
        "prereg_path": str(prereg_path),
        "prereg": prereg_status,
    }


def check_telelogs_agent(data_dir: str | Path, prereg_path: str | Path) -> dict[str, Any]:
    prereg_path = Path(prereg_path)
    base = {
        "source": _TELELOGS_AGENT_SOURCE["source"],
        "repo_id": _TELELOGS_AGENT_SOURCE["repo_id"],
        "download_command": _TELELOGS_AGENT_SOURCE["download_command"],
        "latest_download_attempt": _latest_json(_TELELOGS_AGENT_SOURCE["download_attempt_glob"]),
    }
    try:
        dataset = TeleLogsAgentDataset(data_dir)
    except TeleLogsAgentDatasetError as exc:
        return {
            **base,
            "status": "missing_data",
            "ready_for_headline": False,
            "ready_for_fallback": False,
            "reason": str(exc),
            "data_dir": str(Path(data_dir)),
            "prereg_path": str(prereg_path),
            "runner_modes": _TELELOGS_RUNNER_MODES,
        }

    manifest = _telelogs_manifest(dataset)
    current = {
        "counts": dataset.counts(),
        "manifest_sha256": manifest["sha256"],
        "total_bytes": manifest["total_bytes"],
    }
    prereg = _load_prereg(prereg_path)
    prereg_status = _compare_telelogs_prereg(dataset, current, prereg)
    return {
        **base,
        "status": "ready_fallback" if prereg_status["matches"] else "needs_frozen_prereg",
        "ready_for_headline": False,
        "ready_for_fallback": bool(prereg_status["matches"]),
        "data_dir": str(dataset.root_dir),
        "current": current,
        "prereg_path": str(prereg_path),
        "prereg": prereg_status,
        "runner_modes": _TELELOGS_RUNNER_MODES,
    }


def check_netop_dataset(key: str, data_dir: str | Path | None = None) -> dict[str, Any]:
    config = _NETOP_DATASETS[key]
    root = Path(data_dir or config["data_dir"])
    files = _candidate_files(root, config["expected_globs"])
    base = {
        "ready_for_headline": False,
        "ready_for_fallback": False,
        "source": config["source"],
        "repo_id": config["repo_id"],
        "data_dir": str(root),
        "role_in_claim": config["role"],
        "download_command": f"scripts/download_netop_telco_dataset.sh {key}",
    }
    if not root.exists():
        return {
            **base,
            "status": "missing_data",
            "reason": f"{config['name']} directory does not exist: {root}",
            "access_note": "Hugging Face gated dataset; accept terms and provide HF_TOKEN/HUGGING_FACE_HUB_TOKEN.",
        }
    if not files:
        return {
            **base,
            "status": "missing_data",
            "reason": f"No candidate JSON/CSV files found under {root}",
            "access_note": "Hugging Face gated dataset; accept terms and rerun the download command.",
        }
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        **base,
        "status": "needs_adapter",
        "reason": "Data-like files exist locally, but no label-safe runner/evaluator is registered yet.",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sample_files": [str(path.relative_to(root)) for path in files[:20]],
    }


def check_official_telelogs(data_dir: str | Path, prereg_path: str | Path) -> dict[str, Any]:
    config = _NETOP_DATASETS["telelogs"]
    prereg_path = Path(prereg_path)
    try:
        dataset = TeleLogsDataset(data_dir)
    except TeleLogsDatasetError as exc:
        return {
            "status": "missing_data",
            "ready_for_headline": False,
            "ready_for_fallback": False,
            "source": config["source"],
            "repo_id": config["repo_id"],
            "data_dir": str(Path(data_dir)),
            "prereg_path": str(prereg_path),
            "role_in_claim": config["role"],
            "download_command": "scripts/download_netop_telco_dataset.sh telelogs",
            "reason": str(exc),
            "access_note": "Hugging Face gated dataset; accept terms and provide HF_TOKEN/HUGGING_FACE_HUB_TOKEN.",
        }
    manifest = _official_telelogs_manifest(dataset)
    current = {
        "splits": list(dataset.splits),
        "counts": dataset.counts(),
        "manifest_sha256": manifest["sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }
    prereg = _load_prereg(prereg_path)
    prereg_status = _compare_official_telelogs_prereg(dataset, current, prereg)
    return {
        "status": "ready_fallback" if prereg_status["matches"] else "needs_frozen_prereg",
        "ready_for_headline": False,
        "ready_for_fallback": bool(prereg_status["matches"]),
        "source": config["source"],
        "repo_id": config["repo_id"],
        "data_dir": str(dataset.root_dir),
        "prereg_path": str(prereg_path),
        "role_in_claim": config["role"],
        "download_command": "scripts/download_netop_telco_dataset.sh telelogs",
        "current": current,
        "prereg": prereg_status,
        "runner_modes": {
            "profile": "ingestion/scoring smoke only",
            "llm": "label-safe staged LLM prompt over stripped TeleLogs task JSON",
        },
    }


def check_telecomts(data_dir: str | Path, prereg_path: str | Path) -> dict[str, Any]:
    prereg_path = Path(prereg_path)
    base = {
        **_TELECOMTS_SOURCE,
        "ready_for_headline": False,
        "ready_for_fallback": False,
        "ready_for_synthetic_fallback": False,
        "data_dir": str(Path(data_dir)),
        "prereg_path": str(prereg_path),
        "evidence_tier": "public_5g_testbed_backed_synthetic_rca",
        "role_in_claim": "synthetic-only fallback; not real-fault or operator-network RCA",
        "upstream_protocol_note": (
            "The published RCA task excludes real jamming and classifies ten synthetic anomaly "
            "types injected into measured 5G testbed KPI windows."
        ),
        "local_support": {
            "adapter": True,
            "preregistration": True,
            "runner": True,
            "result_analysis": True,
        },
    }
    try:
        dataset = TelecomTSDataset(data_dir)
        manifest = _telecomts_manifest(dataset)
        training_catalog = build_training_catalog(dataset)
        current = {
            "manifest_sha256": manifest["sha256"],
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "counts": dataset.counts(),
            "class_counts": dataset.class_counts(),
            "event_counts": dataset.event_counts(),
            "class_event_counts": dataset.class_event_counts(),
            "source_sessions": list(dataset.source_sessions),
            "complete_official_layout": dataset.complete_official_layout,
            "training_catalog_policy_id": CATALOG_POLICY_ID,
            "training_catalog_sha256": catalog_sha256(training_catalog),
        }
    except (TelecomTSDatasetError, ValueError) as exc:
        return {
            **base,
            "status": "missing_data",
            "reason": str(exc),
        }

    prereg = _load_prereg(prereg_path)
    prereg_status = _compare_telecomts_prereg(dataset, current, prereg)
    if not dataset.complete_official_layout:
        status = "incomplete_data"
        reason = "TelecomTS does not contain all nine official synthetic source sessions."
    elif not prereg_status["matches"]:
        status = "needs_frozen_prereg"
        reason = "TelecomTS data are present, but no matching frozen test preregistration exists."
    else:
        status = "ready_synthetic_fallback"
        reason = (
            "TelecomTS event-level runner and frozen preregistration are ready for an explicitly "
            "synthetic-only testbed-backed RCA evaluation."
        )
    return {
        **base,
        "status": status,
        "reason": reason,
        "ready_for_synthetic_fallback": status == "ready_synthetic_fallback",
        "current": current,
        "prereg": prereg_status,
    }


def check_synthetic_telco(path: str | Path, prereg_path: str | Path) -> dict[str, Any]:
    path = Path(path)
    prereg_path = Path(prereg_path)
    if not path.exists():
        return {
            "status": "missing_artifact",
            "ready_for_headline": False,
            "ready_for_fallback": False,
            "ready_for_synthetic_fallback": False,
            "reason": f"Synthetic telco dataset artifact does not exist: {path}",
            "path": str(path),
            "prereg_path": str(prereg_path),
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid_json",
            "ready_for_headline": False,
            "ready_for_fallback": False,
            "ready_for_synthetic_fallback": False,
            "reason": str(exc),
            "path": str(path),
            "prereg_path": str(prereg_path),
        }
    validation = validate_synthetic_telco(data)
    current = {
        "sha256": _sha256_synthetic_file(path),
        "content_sha256": _canonical_sha256(data.get("cases", [])),
        "committed_content_sha256": data.get("meta", {}).get("content_sha256"),
        "suite": data.get("meta", {}).get("suite"),
        "seed_source": (data.get("meta", {}).get("design") or {}).get("seed_source"),
        "case_count": validation["case_count"],
        "runtime_case_ids": [
            case.get("runtime", {}).get("runtime_case_id")
            for case in data.get("cases", [])
            if case.get("runtime", {}).get("runtime_case_id")
        ],
        "source_scenario_ids": [
            case.get("labels", {}).get("source_scenario_id")
            for case in data.get("cases", [])
            if case.get("labels", {}).get("source_scenario_id")
        ],
    }
    prereg = _load_prereg(prereg_path)
    prereg_status = _compare_synthetic_prereg(current, prereg)
    matches = validation["ok"] and prereg_status["matches"]
    return {
        "status": "ready_synthetic_fallback" if matches else ("needs_frozen_prereg" if validation["ok"] else "invalid_artifact"),
        "ready_for_headline": False,
        "ready_for_fallback": False,
        "ready_for_synthetic_fallback": bool(matches),
        "path": str(path),
        "prereg_path": str(prereg_path),
        "case_count": validation["case_count"],
        "suite": current["suite"],
        "current": current,
        "validation": validation,
        "prereg": prereg_status,
        "role_in_claim": data.get("meta", {}).get("role_in_claim", "synthetic fallback only"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check benchmark data/prereg readiness.")
    parser.add_argument("--openrca-data-dir", default="data/openrca")
    parser.add_argument("--openrca-prereg", default="results/prereg_openrca_telecom_frozen.json")
    parser.add_argument("--telelogs-data-dir", default="data/telelogs_agent")
    parser.add_argument("--telelogs-prereg", default="results/prereg_telelogs_agent_frozen.json")
    parser.add_argument("--official-telelogs-data-dir", default="data/telelogs")
    parser.add_argument("--official-telelogs-prereg", default="results/prereg_telelogs_frozen.json")
    parser.add_argument("--telecomts-data-dir", default="data/telecomts")
    parser.add_argument("--telecomts-prereg", default="results/prereg_telecomts_frozen.json")
    parser.add_argument("--synthetic-telco", default="results/synthetic_telco_v3_dataset.json")
    parser.add_argument("--synthetic-telco-prereg", default="results/prereg_synthetic_telco_v3_frozen.json")
    parser.add_argument("--out", default="results/benchmark_readiness.json")
    parser.add_argument("--strict", action="store_true", help="exit 2 unless a headline, gated fallback, or synthetic fallback benchmark is ready")
    args = parser.parse_args(argv)

    report = build_readiness_report(
        openrca_data_dir=args.openrca_data_dir,
        openrca_prereg=args.openrca_prereg,
        telelogs_data_dir=args.telelogs_data_dir,
        telelogs_prereg=args.telelogs_prereg,
        official_telelogs_data_dir=args.official_telelogs_data_dir,
        official_telelogs_prereg=args.official_telelogs_prereg,
        telecomts_data_dir=args.telecomts_data_dir,
        telecomts_prereg=args.telecomts_prereg,
        synthetic_telco_path=args.synthetic_telco,
        synthetic_telco_prereg=args.synthetic_telco_prereg,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "headline_ready": report["headline_ready"],
        "fallback_ready": report["fallback_ready"],
        "synthetic_ready": report["synthetic_ready"],
        "next_action": report["next_action"],
    }, indent=2))
    if args.strict and not (report["headline_ready"] or report["fallback_ready"] or report["synthetic_ready"]):
        return 2
    return 0


def _check_tn_rca530() -> dict[str, Any]:
    return {
        "status": "source_only_no_artifact",
        "artifact_status": "no_official_download_url_configured",
        "ready_for_headline": False,
        "ready_for_fallback": False,
        "reason": (
            "TN-RCA530 is a strong source-level headline candidate, but no official dataset "
            "download URL, local data path, or adapter schema is configured yet."
        ),
        "sources": [
            "https://arxiv.org/html/2507.18190v1",
            "https://arxiv.org/abs/2507.18190",
            "https://openreview.net/forum?id=s5mwg63B02",
            "https://huggingface.co/papers/2507.18190",
        ],
        "source_refresh_date": "2026-07-04",
        "benchmark_claims": {
            "scenario_count": 530,
            "domain": "real-world telecommunication alarm RCA",
            "representation": "expert-validated knowledge graphs with topology, equipment, and alarm data",
            "task": "identify ground-truth root-cause tuple(s) from candidate graph evidence",
            "primary_metric": "macro F1 over exact root-cause tuples",
            "difficulty_note": "paper reports a long-tail distribution and mostly difficult scenarios",
        },
        "local_support": {
            "adapter": False,
            "data_dir": None,
            "preregistration": False,
        },
        "next_action": (
            "Obtain an official TN-RCA530 release URL or files; implement a telco_mas.tnrca "
            "adapter only after the released schema is available."
        ),
    }


def _candidate_files(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.name != "README.md" and path not in seen:
                seen.add(path)
                out.append(path)
    return out


def _latest_json(pattern: str) -> dict[str, Any] | None:
    paths = sorted(Path().glob(pattern))
    if not paths:
        return None
    path = paths[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(path), "status": "unreadable", "reason": str(exc)}
    if isinstance(payload, dict):
        payload = {**payload}
        payload.setdefault("path", str(path))
        return payload
    return {"path": str(path), "status": "invalid_json_shape"}


def _compare_openrca_prereg(dataset: OpenRCADataset, current: dict[str, Any], prereg: dict[str, Any] | None) -> dict[str, Any]:
    if prereg is None:
        return {"exists": False, "matches": False, "reason": "missing preregistration JSON"}
    expected = prereg.get("dataset", {})
    row_selection = prereg.get("row_selection", {}) if isinstance(prereg.get("row_selection"), dict) else {}
    rows = row_selection.get("row_ids", [])
    if "confirmatory_row_count" in row_selection:
        confirmatory_row_count = int(row_selection.get("confirmatory_row_count") or 0)
    else:
        confirmatory_row_count = len(rows) if isinstance(rows, list) else 0
    row_ids_valid = isinstance(rows, list) and all(isinstance(row_id, int) and 0 <= row_id < len(dataset.rows) for row_id in rows)
    prepared = expected.get("prepared") if isinstance(expected.get("prepared"), dict) else {}
    prepared_path = Path(str(prepared.get("manifest_path") or ""))
    prepared_matches = (
        bool(
            prepared_path.is_file()
            and prepared.get("manifest_sha256") == _sha256_file(prepared_path)
        )
        if prepared
        else True
    )
    roles = prereg.get("system_roles", {})
    role_protocol_valid = (
        (
            roles.get("architecture_baseline") == ["rca_agent_replica"]
            and roles.get("diagnostic_oracle") == ["same_board_single"]
            and roles.get("operational_single") == ["single_react_sc"]
            and roles.get("treatment") == "shardrca_full"
        )
        if roles
        else True
    )
    matches = (
        prereg.get("status") == "frozen"
        and expected.get("query_sha256") == current["query_sha256"]
        and expected.get("telemetry_manifest_sha256") == current["telemetry_manifest_sha256"]
        and row_ids_valid
        and bool(rows)
        and confirmatory_row_count > 0
        and prepared_matches
        and role_protocol_valid
    )
    reason = None
    if not matches:
        if confirmatory_row_count <= 0:
            reason = "matching preregistration has zero confirmatory rows after contamination ledger"
        elif prereg.get("status") != "frozen":
            reason = "preregistration is not frozen"
        elif expected.get("query_sha256") != current["query_sha256"]:
            reason = "query.csv differs from preregistration"
        elif expected.get("telemetry_manifest_sha256") != current["telemetry_manifest_sha256"]:
            reason = "telemetry manifest differs from preregistration"
        elif not row_ids_valid or not rows:
            reason = "row selection is missing or invalid"
        elif not prepared_matches:
            reason = "prepared manifest differs from preregistration"
        elif not role_protocol_valid:
            reason = "OpenRCA system-role protocol is incomplete"
    return {
        "exists": True,
        "matches": matches,
        "reason": reason,
        "status": prereg.get("status"),
        "query_sha256_matches": expected.get("query_sha256") == current["query_sha256"],
        "telemetry_manifest_matches": expected.get("telemetry_manifest_sha256") == current["telemetry_manifest_sha256"],
        "row_ids_valid": row_ids_valid,
        "prepared_manifest_matches": prepared_matches,
        "role_protocol_valid": role_protocol_valid,
        "row_count": len(rows) if isinstance(rows, list) else 0,
        "confirmatory_row_count": confirmatory_row_count,
    }


def _compare_telelogs_prereg(
    dataset: TeleLogsAgentDataset,
    current: dict[str, Any],
    prereg: dict[str, Any] | None,
) -> dict[str, Any]:
    if prereg is None:
        return {"exists": False, "matches": False, "reason": "missing preregistration JSON"}
    expected = prereg.get("dataset", {})
    selected = prereg.get("row_selection", {}).get("selected", {})
    selected_valid = _selected_valid(dataset, selected)
    matches = (
        prereg.get("status") == "frozen"
        and expected.get("manifest_sha256") == current["manifest_sha256"]
        and expected.get("counts") == current["counts"]
        and selected_valid
        and any(selected.values()) if isinstance(selected, dict) else False
    )
    return {
        "exists": True,
        "matches": bool(matches),
        "status": prereg.get("status"),
        "manifest_matches": expected.get("manifest_sha256") == current["manifest_sha256"],
        "counts_match": expected.get("counts") == current["counts"],
        "selected_valid": selected_valid,
        "selected_total": sum(len(ids) for ids in selected.values()) if isinstance(selected, dict) else 0,
    }


def _compare_official_telelogs_prereg(
    dataset: TeleLogsDataset,
    current: dict[str, Any],
    prereg: dict[str, Any] | None,
) -> dict[str, Any]:
    if prereg is None:
        return {"exists": False, "matches": False, "reason": "missing preregistration JSON"}
    expected = prereg.get("dataset", {})
    selection = prereg.get("row_selection", {})
    split = selection.get("split")
    row_ids = selection.get("row_ids", [])
    row_ids_valid = (
        isinstance(split, str)
        and split in dataset.splits
        and isinstance(row_ids, list)
        and all(isinstance(row_id, int) and 0 <= row_id < len(dataset.rows(split)) for row_id in row_ids)
    )
    matches = (
        prereg.get("status") == "frozen"
        and expected.get("manifest_sha256") == current["manifest_sha256"]
        and expected.get("counts") == current["counts"]
        and row_ids_valid
        and bool(row_ids)
        and split == "test"
    )
    return {
        "exists": True,
        "matches": bool(matches),
        "status": prereg.get("status"),
        "manifest_matches": expected.get("manifest_sha256") == current["manifest_sha256"],
        "counts_match": expected.get("counts") == current["counts"],
        "row_ids_valid": row_ids_valid,
        "split": split,
        "official_test_split": split == "test",
        "row_count": len(row_ids) if isinstance(row_ids, list) else 0,
    }


def _compare_synthetic_prereg(current: dict[str, Any], prereg: dict[str, Any] | None) -> dict[str, Any]:
    if prereg is None:
        return {"exists": False, "matches": False, "reason": "missing preregistration JSON"}
    expected = prereg.get("dataset", {})
    selected = prereg.get("row_selection", {}).get("runtime_case_ids", [])
    ids_match = selected == current["runtime_case_ids"]
    source_ids = prereg.get("row_selection", {}).get("source_scenario_ids", [])
    source_ids_match = source_ids == current["source_scenario_ids"]
    content_matches = (
        current["content_sha256"] == current["committed_content_sha256"]
        and expected.get("content_sha256") == current["content_sha256"]
    )
    v4_protocol = current.get("suite") != "telco_v4" or (
        bool(current.get("seed_source"))
        and source_ids_match
        and bool(prereg.get("algorithm", {}).get("id"))
        and prereg.get("algorithm", {}).get("id")
        != "development artifact; not a locked confirmatory algorithm"
        and int(prereg.get("execution", {}).get("runs") or 0) >= 1
    )
    matches = (
        prereg.get("status") == "frozen"
        and expected.get("sha256") == current["sha256"]
        and content_matches
        and expected.get("case_count") == current["case_count"]
        and ids_match
        and bool(selected)
        and v4_protocol
    )
    return {
        "exists": True,
        "matches": matches,
        "status": prereg.get("status"),
        "sha256_matches": expected.get("sha256") == current["sha256"],
        "content_sha256_matches": content_matches,
        "case_count_matches": expected.get("case_count") == current["case_count"],
        "runtime_case_ids_match": ids_match,
        "source_scenario_ids_match": source_ids_match,
        "v4_protocol_complete": v4_protocol,
        "case_count": len(selected) if isinstance(selected, list) else 0,
    }


def _compare_telecomts_prereg(
    dataset: TelecomTSDataset,
    current: dict[str, Any],
    prereg: dict[str, Any] | None,
) -> dict[str, Any]:
    if prereg is None:
        return {"exists": False, "matches": False, "reason": "missing preregistration JSON"}
    expected = prereg.get("dataset", {})
    selection = prereg.get("event_selection", {})
    event_indices = selection.get("event_indices", [])
    event_ids = selection.get("event_ids", [])
    split = selection.get("split")
    events = dataset.events(split) if split in dataset.splits else []
    event_indices_valid = (
        isinstance(event_indices, list)
        and bool(event_indices)
        and len(set(event_indices)) == len(event_indices)
        and all(
            isinstance(event_index, int) and 0 <= event_index < len(events)
            for event_index in event_indices
        )
    )
    selected_event_ids = (
        [events[event_index].event_id for event_index in event_indices]
        if event_indices_valid
        else []
    )
    event_ids_match = event_ids == selected_event_ids
    algorithm = prereg.get("algorithm", {})
    expected_catalog = prereg.get("training_catalog", {})
    catalog_matches = (
        expected_catalog.get("policy_id") == current["training_catalog_policy_id"]
        and expected_catalog.get("sha256") == current["training_catalog_sha256"]
        and expected_catalog.get("source_split") == "development"
        and expected_catalog.get("uses_validation_or_test_labels") is False
    )
    algorithm_locked = (
        bool(algorithm.get("locked_before_test"))
        and bool(algorithm.get("id"))
        and not str(algorithm.get("id")).startswith("unlocked draft")
    )
    matches = (
        prereg.get("status") == "frozen"
        and expected.get("manifest_sha256") == current["manifest_sha256"]
        and expected.get("counts") == current["counts"]
        and expected.get("event_counts") == current["event_counts"]
        and expected.get("source_sessions") == current["source_sessions"]
        and expected.get("complete_official_layout") is True
        and catalog_matches
        and split == "test"
        and event_indices_valid
        and event_ids_match
        and algorithm_locked
    )
    return {
        "exists": True,
        "matches": bool(matches),
        "status": prereg.get("status"),
        "manifest_matches": expected.get("manifest_sha256") == current["manifest_sha256"],
        "counts_match": expected.get("counts") == current["counts"],
        "event_counts_match": expected.get("event_counts") == current["event_counts"],
        "source_sessions_match": expected.get("source_sessions") == current["source_sessions"],
        "complete_official_layout": expected.get("complete_official_layout") is True,
        "training_catalog_matches": catalog_matches,
        "official_test_split": split == "test",
        "event_indices_valid": event_indices_valid,
        "event_ids_match": event_ids_match,
        "algorithm_locked": algorithm_locked,
        "event_count": len(event_indices) if isinstance(event_indices, list) else 0,
    }


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _selected_valid(dataset: TeleLogsAgentDataset, selected: Any) -> bool:
    if not isinstance(selected, dict):
        return False
    for name, row_ids in selected.items():
        if name not in dataset.scenario_sets or not isinstance(row_ids, list):
            return False
        row_count = len(dataset.rows(name))
        if any(not isinstance(row_id, int) or row_id < 0 or row_id >= row_count for row_id in row_ids):
            return False
    return True


def _load_prereg(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _next_action(
    headline_ready: bool,
    fallback_ready: bool,
    synthetic_ready: bool,
    benchmarks: dict[str, dict[str, Any]],
) -> str:
    if headline_ready:
        return "Run the frozen real benchmark systems exactly as preregistered."
    if benchmarks["openrca_telecom"]["status"] == "needs_frozen_prereg":
        if benchmarks["openrca_telecom"].get("prereg", {}).get("confirmatory_row_count") == 0:
            return "Acquire or freeze a clean OpenRCA/TN-RCA530 run with non-contaminated confirmatory rows."
        return "Generate/fix OpenRCA preregistration before running live systems."
    if fallback_ready:
        return "Real benchmark is not ready; fallback TeleLogsAgent can be run with explicit synthetic limitation."
    if benchmarks["telelogs_agent"]["status"] == "needs_frozen_prereg":
        return "Generate/fix TeleLogsAgent preregistration if using synthetic fallback."
    access_action = _data_access_next_action(benchmarks)
    telecomts_action = _telecomts_next_action(benchmarks)
    if access_action:
        if synthetic_ready:
            return (
                f"{access_action} {telecomts_action} Local synthetic telco data are ready only as development/synthetic "
                "fallback, not headline evidence."
            ).strip()
        return f"{access_action} {telecomts_action}".strip()
    if synthetic_ready:
        suite = benchmarks["synthetic_telco"].get("suite")
        if suite == "telco_v4":
            return (
                "Frozen telco-v4 protocol is technically ready; run it only for the locked "
                "confirmatory evaluation, disclose synthetic limitations, and keep pursuing real data."
            )
        return (
            "Only development-only synthetic telco data are ready; do not call them an unseen "
            "confirmatory result, and keep pursuing real/gated data."
        )
    if benchmarks["synthetic_telco"]["status"] == "needs_frozen_prereg":
        return "Generate/fix the synthetic telco preregistration before any local synthetic fallback run."
    return "Acquire OpenRCA Telecom or TN-RCA530 data; TeleLogsAgent is also missing fallback files."


def _telecomts_next_action(benchmarks: dict[str, dict[str, Any]]) -> str:
    telecomts = benchmarks.get("telecomts", {})
    status = telecomts.get("status")
    if status == "missing_data":
        return "TelecomTS is publicly downloadable with `scripts/download_telecomts.sh`."
    if status == "incomplete_data":
        return "Redownload the complete TelecomTS anomaly subset."
    if status == "needs_frozen_prereg":
        return (
            "TelecomTS runner is local; run only development/validation with a live model, "
            "then lock the algorithm before freezing test preregistration."
        )
    if status == "ready_synthetic_fallback":
        return "Run TelecomTS exactly as frozen and label any result synthetic-only."
    return ""


def _data_access_next_action(benchmarks: dict[str, dict[str, Any]]) -> str:
    actions = []
    telelogs_agent = benchmarks.get("telelogs_agent", {})
    if telelogs_agent.get("status") == "missing_data":
        attempt = telelogs_agent.get("latest_download_attempt")
        attempt_status = attempt.get("status") if isinstance(attempt, dict) else None
        if attempt_status == "blocked_token_not_authorized":
            actions.append(
                "Wait for Hugging Face approval for TeleLogsAgent, then rerun "
                "`scripts/download_telelogs_agent.sh` and freeze preregistration."
            )
        elif attempt_status == "blocked_gated_access":
            actions.append(
                "Accept TeleLogsAgent gated terms and rerun `scripts/download_telelogs_agent.sh` "
                "with an authorized HF token."
            )
        else:
            actions.append(
                "Obtain TeleLogsAgent TS1/TS2/TS3 files, then freeze preregistration."
            )

    openrca = benchmarks.get("openrca_telecom", {})
    if openrca.get("status") == "missing_data":
        attempt = openrca.get("latest_download_attempt")
        attempt_status = attempt.get("status") if isinstance(attempt, dict) else None
        if attempt_status == "blocked_google_drive_quota":
            actions.append(
                "Retry or manually download OpenRCA Telecom after the Google Drive quota clears."
            )
        else:
            actions.append(
                "Download and extract OpenRCA Telecom under `data/openrca/Telecom`."
            )

    tn_rca = benchmarks.get("tn_rca530", {})
    if tn_rca.get("status") == "source_only_no_artifact":
        actions.append("Keep searching for an official TN-RCA530 artifact/release URL.")

    return " ".join(actions)


if __name__ == "__main__":
    raise SystemExit(main())

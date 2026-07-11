"""Readiness checks for the paper-facing RCA benchmarks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..openrca.dataset import OpenRCADataset, OpenRCADatasetError
from ..openrca.prereg import _sha256_file, _telemetry_manifest

_OPENRCA_TELECOM_SOURCE = {
    "source": "https://github.com/microsoft/OpenRCA",
    "google_drive_file_id": "1cyOKpqyAP4fy-QiJ6a_cKuwR7D46zyVe",
    "download_command": "bash scripts/download_openrca_telecom.sh --extract",
    "download_attempt_glob": "results/openrca_telecom_download_attempt_*.json",
}


def build_readiness_report(
    *,
    openrca_data_dir: str | Path = "data/openrca",
    openrca_prereg: str | Path = "results/prereg_openrca_telecom_frozen.json",
) -> dict[str, Any]:
    benchmarks = {
        "tn_rca530": _check_tn_rca530(),
        "openrca_telecom": check_openrca(openrca_data_dir, openrca_prereg),
    }
    headline_ready = any(item["ready_for_headline"] for item in benchmarks.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "paper-only ShardRCA/RCAEval/OpenRCA readiness",
        "headline_ready": headline_ready,
        "fallback_ready": False,
        "synthetic_ready": False,
        "benchmarks": benchmarks,
        "next_action": _next_action(headline_ready, benchmarks),
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
    prereg_status = _compare_openrca_prereg(dataset, current, _load_prereg(prereg_path))
    return {
        **base,
        "status": "ready" if prereg_status["matches"] else "needs_frozen_prereg",
        "ready_for_headline": bool(prereg_status["matches"]),
        "ready_for_fallback": False,
        "reason": None if prereg_status["matches"] else prereg_status.get("reason"),
        "data_dir": str(dataset.root_dir),
        "query_path": str(dataset.query_path),
        "telemetry_dir": str(dataset.telemetry_dir),
        "current": current,
        "prereg_path": str(prereg_path),
        "prereg": prereg_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check paper benchmark data/prereg readiness.")
    parser.add_argument("--openrca-data-dir", default="data/openrca")
    parser.add_argument("--openrca-prereg", default="results/prereg_openrca_telecom_frozen.json")
    parser.add_argument("--out", default="results/benchmark_readiness.json")
    parser.add_argument("--strict", action="store_true", help="exit 2 unless a headline benchmark is ready")
    args = parser.parse_args(argv)

    report = build_readiness_report(
        openrca_data_dir=args.openrca_data_dir,
        openrca_prereg=args.openrca_prereg,
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
    if args.strict and not report["headline_ready"]:
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
        },
        "local_support": {"adapter": False, "data_dir": None, "preregistration": False},
        "next_action": "Obtain an official TN-RCA530 release URL or files before adding an adapter.",
    }


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
        return {**payload, "path": str(path)}
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
        bool(prepared_path.is_file() and prepared.get("manifest_sha256") == _sha256_file(prepared_path))
        if prepared
        else True
    )
    roles = prereg.get("system_roles", {})
    role_protocol_valid = (
        (
            roles.get("architecture_baseline") == ["rca_agent_replica"]
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


def _load_prereg(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _next_action(headline_ready: bool, benchmarks: dict[str, dict[str, Any]]) -> str:
    if headline_ready:
        return "Run the frozen OpenRCA/RCAEval analysis and claim audit."
    openrca = benchmarks["openrca_telecom"]
    messages = []
    attempt = openrca.get("latest_download_attempt")
    if isinstance(attempt, dict):
        status = str(attempt.get("status") or "")
        reason = str(attempt.get("reason") or "")
        if "quota" in status.lower() or "quota" in reason.lower():
            messages.append("Resolve the OpenRCA Google Drive quota/download issue.")
    messages.append("Prepare OpenRCA data and freeze a matching preregistration.")
    messages.append("Obtain an official TN-RCA530 artifact if it becomes available.")
    return " ".join(messages)


if __name__ == "__main__":
    raise SystemExit(main())

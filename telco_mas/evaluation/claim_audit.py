"""Audit whether current paper artifacts justify a ShardRCA claim."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

HEADLINE_BENCHMARKS = {"openrca_telecom", "tn_rca530"}
SUPPORTING_ONLY_BENCHMARKS = {"rcaeval"}


def audit_claim(
    *,
    readiness_path: str | Path = "results/benchmark_readiness.json",
    analysis_paths: list[str | Path] | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    del allow_synthetic
    readiness = _load_json(readiness_path)
    analyses = [_audit_analysis(path, readiness) for path in (analysis_paths or [])]
    valid = [item for item in analyses if item["valid_evidence"]]
    headline = [item for item in valid if item["benchmark"] in HEADLINE_BENCHMARKS]
    claim_allowed = bool(headline)
    selected = headline[0] if headline else None

    blockers = _readiness_blockers(readiness)
    if not analysis_paths:
        blockers.append("no result analysis artifacts supplied")
    if analyses and not valid:
        blockers.append("no analysis artifact passed claim-evidence validation")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "claim_allowed": claim_allowed,
        "claim_tier": "headline_real_or_operational" if claim_allowed else "none",
        "selected_evidence": selected,
        "readiness": {
            "headline_ready": bool(readiness.get("headline_ready")),
            "fallback_ready": False,
            "synthetic_ready": False,
            "next_action": readiness.get("next_action"),
        },
        "analysis_audits": analyses,
        "blockers": [] if claim_allowed else blockers,
        "required_for_claim": [
            "OpenRCA or TN-RCA530 readiness is true for the claimed headline tier",
            "paired result analysis clear_win_gate.passed is true",
            "analysis baseline was resolved by --baseline strongest_single",
            "treatment system is shardrca_full/full/multi_agent",
            "overfit_guard.passed is true when present",
            "OpenRCA evidence includes no_interaction/no_topology/no_falsifier/no_refinement ablations and default no-fit weights",
            "runtime candidate catalogs are not label-derived",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether result artifacts justify a ShardRCA win claim.")
    parser.add_argument("--readiness", default="results/benchmark_readiness.json")
    parser.add_argument("--analysis", action="append", default=[], help="result analysis JSON path; repeatable")
    parser.add_argument("--allow-synthetic", action="store_true", help="accepted for CLI compatibility; ignored")
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true", help="exit 2 unless claim_allowed is true")
    args = parser.parse_args(argv)

    payload = audit_claim(
        readiness_path=args.readiness,
        analysis_paths=args.analysis,
        allow_synthetic=args.allow_synthetic,
    )
    text = json.dumps(payload, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(json.dumps({
            "out": str(out),
            "claim_allowed": payload["claim_allowed"],
            "claim_tier": payload["claim_tier"],
        }, indent=2))
    else:
        print(text)
    if args.strict and not payload["claim_allowed"]:
        return 2
    return 0


def _audit_analysis(path: str | Path, readiness: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json(path)
    benchmark = _infer_benchmark(path, payload)
    gate = payload.get("clear_win_gate") if isinstance(payload.get("clear_win_gate"), dict) else {}
    overfit_guard = payload.get("overfit_guard") if isinstance(payload.get("overfit_guard"), dict) else {}
    baseline_selection = payload.get("baseline_selection") if isinstance(payload.get("baseline_selection"), dict) else {}
    treatment = str(payload.get("treatment") or gate.get("treatment") or "")
    enforce_openrca_protocol = benchmark == "openrca_telecom"
    checks = {
        "gate_passed": bool(gate.get("passed")),
        "strongest_single_baseline": str(baseline_selection.get("method") or "").startswith("strongest_single"),
        "mas_treatment": _is_mas_treatment(treatment),
        "tier_ready": _tier_ready(readiness, benchmark),
        "candidate_catalog_not_label_derived": not bool(payload.get("label_derived_candidate_catalog")),
    }
    if overfit_guard:
        checks["overfit_guard_passed"] = bool(overfit_guard.get("passed"))
    if enforce_openrca_protocol:
        comparisons = payload.get("comparisons") if isinstance(payload.get("comparisons"), dict) else {}
        checks.update({
            "protocol_complete": bool(gate.get("protocol_complete")),
            "overfit_guard_passed": bool(overfit_guard.get("passed")) and bool(
                gate.get("overfit_guard_passed", overfit_guard.get("passed"))
            ),
            "rca_agent_comparison": "architecture_baseline" in comparisons,
            "operational_single_comparison": "operational_single" in comparisons,
        })

    reasons = []
    if not checks["gate_passed"]:
        reasons.append("clear_win_gate.passed is false or missing")
    if not checks["strongest_single_baseline"]:
        reasons.append("analysis was not run with --baseline strongest_single")
    if not checks["mas_treatment"]:
        reasons.append(f"treatment is not recognized as MAS: {treatment or 'missing'}")
    if not checks["tier_ready"]:
        if benchmark in SUPPORTING_ONLY_BENCHMARKS:
            reasons.append(f"{benchmark} is supporting-only evidence, not a headline claim tier")
        else:
            reasons.append(f"benchmark readiness is false for {benchmark}")
    if not checks["candidate_catalog_not_label_derived"]:
        reasons.append("candidate catalog is marked label-derived; runtime candidate universes must not come from scoring labels")
    if "overfit_guard_passed" in checks and not checks["overfit_guard_passed"]:
        guard_reasons = overfit_guard.get("reasons") if isinstance(overfit_guard.get("reasons"), list) else []
        suffix = f": {'; '.join(str(item) for item in guard_reasons)}" if guard_reasons else ""
        reasons.append(
            "overfit guard failed or is missing; require no fitted weights, registered mechanism ablations, "
            f"and label-safe candidate catalogs{suffix}"
        )
    if enforce_openrca_protocol:
        if not checks.get("protocol_complete", False):
            reasons.append("OpenRCA result does not contain the complete frozen baseline+treatment+ablation protocol")
        if not checks.get("rca_agent_comparison", False):
            reasons.append("OpenRCA result lacks the RCA-Agent architecture comparison")
        if not checks.get("operational_single_comparison", False):
            reasons.append("OpenRCA result lacks the operational single-agent comparison")

    return {
        "path": str(path),
        "benchmark": benchmark,
        "valid_evidence": all(checks.values()),
        "checks": checks,
        "reasons": reasons,
        "baseline": payload.get("baseline") or gate.get("baseline"),
        "treatment": treatment,
        "clear_win_gate": gate,
        "baseline_selection": baseline_selection,
    }


def _tier_ready(readiness: dict[str, Any], benchmark: str) -> bool:
    if benchmark not in HEADLINE_BENCHMARKS:
        return False
    item = readiness.get("benchmarks", {}).get(benchmark, {})
    return bool(item.get("ready_for_headline"))


def _readiness_blockers(readiness: dict[str, Any]) -> list[str]:
    blockers = []
    for name, item in (readiness.get("benchmarks") or {}).items():
        if item.get("ready_for_headline"):
            continue
        if name in HEADLINE_BENCHMARKS:
            reason = item.get("reason") or item.get("status") or "not ready"
            blockers.append(f"{name}: {reason}{_latest_attempt_note(item)}")
    return blockers


def _latest_attempt_note(item: dict[str, Any]) -> str:
    attempt = item.get("latest_download_attempt")
    if not isinstance(attempt, dict):
        return ""
    status = str(attempt.get("status") or "").strip()
    reason = str(attempt.get("reason") or "").strip()
    next_action = str(attempt.get("next_action") or "").strip()
    bits = []
    if status:
        bits.append(f"latest_attempt={status}")
    if reason:
        bits.append(reason)
    if next_action:
        bits.append(f"next={next_action}")
    return f" [{'; '.join(bits)}]" if bits else ""


def _infer_benchmark(path: Path, payload: dict[str, Any]) -> str:
    explicit = payload.get("benchmark")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    text = " ".join([str(path), " ".join(str(item) for item in payload.get("paths", []))]).lower()
    if "openrca" in text:
        return "openrca_telecom"
    if "rcaeval" in text:
        return "rcaeval"
    return "unknown"


def _is_mas_treatment(treatment: str) -> bool:
    lower = treatment.lower()
    return (
        lower in {"shardrca_full", "full", "multi", "multi_agent"}
        or lower.startswith("shardrca")
        or "_shardrca" in lower
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

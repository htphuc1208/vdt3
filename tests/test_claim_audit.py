import json

from telco_mas.evaluation.claim_audit import audit_claim


def test_claim_audit_rejects_missing_analysis(tmp_path):
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps(_readiness()), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[])

    assert payload["claim_allowed"] is False
    assert "no result analysis artifacts supplied" in payload["blockers"]


def test_claim_audit_rejects_non_strongest_single_analysis(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "openrca_analysis.json"
    readiness.write_text(json.dumps(_readiness(openrca_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        baseline_selection={"method": "explicit", "resolved": "single_react_sc"},
        passed=True,
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is False
    assert payload["analysis_audits"][0]["checks"]["gate_passed"] is True
    assert payload["analysis_audits"][0]["checks"]["strongest_single_baseline"] is False
    assert "no analysis artifact passed claim-evidence validation" in payload["blockers"]


def test_claim_audit_accepts_ready_openrca_strongest_single_win(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "openrca_analysis.json"
    readiness.write_text(json.dumps(_readiness(openrca_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_equal_tokens",
        },
        passed=True,
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is True
    assert payload["claim_tier"] == "headline_real_or_operational"
    assert payload["selected_evidence"]["benchmark"] == "openrca_telecom"
    assert payload["blockers"] == []


def test_claim_audit_rejects_label_derived_candidate_catalog(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "openrca_analysis.json"
    readiness.write_text(json.dumps(_readiness(openrca_ready=True)), encoding="utf-8")
    payload = _analysis(
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_equal_tokens",
        },
        passed=True,
    )
    payload["label_derived_candidate_catalog"] = True
    analysis.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert audit["claim_allowed"] is False
    assert audit["analysis_audits"][0]["checks"]["candidate_catalog_not_label_derived"] is False
    assert any("label-derived" in reason for reason in audit["analysis_audits"][0]["reasons"])


def test_claim_audit_rejects_openrca_failed_overfit_guard(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "openrca_analysis.json"
    readiness.write_text(json.dumps(_readiness(openrca_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_react_sc",
        },
        passed=True,
        overfit_guard_passed=False,
    )), encoding="utf-8")

    audit = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert audit["claim_allowed"] is False
    assert audit["analysis_audits"][0]["checks"]["overfit_guard_passed"] is False
    assert any("overfit guard failed" in reason for reason in audit["analysis_audits"][0]["reasons"])


def test_claim_audit_blockers_include_latest_download_attempt(tmp_path):
    readiness = tmp_path / "readiness.json"
    payload = _readiness()
    payload["benchmarks"]["openrca_telecom"]["latest_download_attempt"] = {
        "status": "blocked_google_drive_quota",
        "reason": "quota",
        "next_action": "retry later",
    }
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_claim(readiness_path=readiness, analysis_paths=[])

    openrca_blocker = next(item for item in audit["blockers"] if item.startswith("openrca_telecom:"))
    assert "latest_attempt=blocked_google_drive_quota" in openrca_blocker
    assert "quota" in openrca_blocker
    assert "next=retry later" in openrca_blocker


def test_claim_audit_rejects_rcaeval_as_supporting_only_even_if_gate_passes(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "rcaeval_analysis.json"
    readiness.write_text(json.dumps(_readiness()), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="rcaeval",
        paths=["results/rcaeval_hard_llm_v7_holdout20.json"],
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "rcaeval_single_react_sc",
        },
        passed=True,
        treatment="rcaeval_shardrca_full",
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is False
    assert payload["analysis_audits"][0]["benchmark"] == "rcaeval"
    assert payload["analysis_audits"][0]["checks"]["tier_ready"] is False
    assert "supporting-only evidence" in payload["analysis_audits"][0]["reasons"][0]


def test_claim_audit_prefers_explicit_benchmark_over_filename(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "openrca_named_but_rcaeval_payload.json"
    readiness.write_text(json.dumps(_readiness(openrca_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="rcaeval",
        paths=["results/openrca_looking_name.json"],
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "rcaeval_single_react_sc",
        },
        passed=True,
        treatment="rcaeval_shardrca_full",
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is False
    assert payload["analysis_audits"][0]["benchmark"] == "rcaeval"
    assert "supporting-only evidence" in payload["analysis_audits"][0]["reasons"][0]


def _readiness(*, openrca_ready: bool = False) -> dict:
    return {
        "headline_ready": openrca_ready,
        "fallback_ready": False,
        "synthetic_ready": False,
        "next_action": "unit test",
        "benchmarks": {
            "openrca_telecom": {
                "ready_for_headline": openrca_ready,
                "ready_for_fallback": False,
                "reason": "" if openrca_ready else "missing data",
            },
            "tn_rca530": {
                "ready_for_headline": False,
                "ready_for_fallback": False,
                "reason": "not integrated",
            },
        },
    }


def _analysis(
    *,
    benchmark: str | None = None,
    paths: list[str] | None = None,
    baseline_selection: dict,
    passed: bool,
    treatment: str = "shardrca_full",
    overfit_guard_passed: bool = True,
) -> dict:
    payload = {
        "paths": paths or ["results/openrca_paired_frozen.json"],
        "baseline": baseline_selection.get("resolved", "single_react_sc"),
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "comparisons": {
            "architecture_baseline": {},
            "operational_single": {},
        },
        "overfit_guard": {
            "passed": overfit_guard_passed,
            "checks": {
                "required_mechanism_ablations": overfit_guard_passed,
                "weights_declared_no_fit": overfit_guard_passed,
                "candidate_catalog_not_label_derived": True,
            },
            "reasons": [] if overfit_guard_passed else ["unit-test failed overfit guard"],
        },
        "clear_win_gate": {
            "passed": passed,
            "protocol_complete": True,
            "overfit_guard_passed": overfit_guard_passed,
            "baseline": baseline_selection.get("resolved", "single_react_sc"),
            "treatment": treatment,
        },
    }
    if benchmark:
        payload["benchmark"] = benchmark
    return payload

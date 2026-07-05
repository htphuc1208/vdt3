import json
from pathlib import Path

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


def test_claim_audit_accepts_ready_official_telco_fallback(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "telelogs_analysis.json"
    readiness.write_text(json.dumps(_readiness(telelogs_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        paths=["results/telelogs_paired_llm.json"],
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_react_sc",
        },
        passed=True,
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is True
    assert payload["claim_tier"] == "official_telco_synthetic_fallback"
    assert payload["selected_evidence"]["benchmark"] == "telelogs"


def test_claim_audit_accepts_ready_telelogs_agent_official_tool_win(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "telelogs_agent_tool_analysis.json"
    readiness.write_text(json.dumps(_readiness(telelogs_agent_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="telelogs_agent",
        paths=["results/telelogs_agent_tool.json"],
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_react_sc",
        },
        passed=True,
        official_tool_mode=True,
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is True
    assert payload["claim_tier"] == "official_telco_synthetic_fallback"
    assert payload["selected_evidence"]["benchmark"] == "telelogs_agent"


def test_claim_audit_rejects_telelogs_agent_staged_mode_even_if_gate_passes(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "telelogs_agent_profile_analysis.json"
    readiness.write_text(json.dumps(_readiness(telelogs_agent_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="telelogs_agent",
        paths=["results/telelogs_agent_profile.json"],
        baseline_selection={
            "method": "strongest_single_by_strict_then_score_then_tokens",
            "resolved": "single_react_sc",
        },
        passed=True,
        official_tool_mode=False,
    )), encoding="utf-8")

    payload = audit_claim(readiness_path=readiness, analysis_paths=[analysis])

    assert payload["claim_allowed"] is False
    audit = payload["analysis_audits"][0]
    assert audit["checks"]["gate_passed"] is True
    assert audit["checks"]["tier_ready"] is True
    assert audit["checks"]["official_tool_mode"] is False
    assert "official HTTP-tool mode" in audit["reasons"][0]
    assert "no analysis artifact passed claim-evidence validation" in payload["blockers"]


def test_claim_audit_blockers_include_latest_download_attempt(tmp_path):
    readiness = tmp_path / "readiness.json"
    payload = _readiness()
    payload["benchmarks"]["telelogs_agent"]["latest_download_attempt"] = {
        "status": "blocked_token_not_authorized",
        "reason": "account is not in the authorized list",
        "next_action": "request dataset access",
    }
    readiness.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_claim(readiness_path=readiness, analysis_paths=[])

    telelogs_blocker = next(item for item in audit["blockers"] if item.startswith("telelogs_agent:"))
    assert "latest_attempt=blocked_token_not_authorized" in telelogs_blocker
    assert "account is not in the authorized list" in telelogs_blocker
    assert "next=request dataset access" in telelogs_blocker


def test_claim_audit_rejects_rcaeval_as_supporting_only_even_if_gate_passes(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "rcaeval_analysis.json"
    readiness.write_text(json.dumps(_readiness()), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
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
    assert payload["analysis_audits"][0]["checks"]["gate_passed"] is True
    assert payload["analysis_audits"][0]["checks"]["strongest_single_baseline"] is True
    assert payload["analysis_audits"][0]["checks"]["mas_treatment"] is True
    assert payload["analysis_audits"][0]["checks"]["tier_ready"] is False
    assert "supporting-only evidence" in payload["analysis_audits"][0]["reasons"][0]
    assert "no analysis artifact passed claim-evidence validation" in payload["blockers"]


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


def test_claim_audit_requires_explicit_synthetic_opt_in_for_telecomts(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "misleading_openrca_name.json"
    readiness.write_text(json.dumps(_readiness(telecomts_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="telecomts",
        baseline_selection={
            "method": "strongest_single_by_macro_then_micro_then_tokens",
            "resolved": "single_equal_calls",
        },
        passed=True,
        treatment="telecomts_shardrca_full",
        llm_mode=True,
        test_split=True,
        event_unit=True,
        frozen_prereg=True,
    )), encoding="utf-8")

    rejected = audit_claim(readiness_path=readiness, analysis_paths=[analysis])
    accepted = audit_claim(
        readiness_path=readiness,
        analysis_paths=[analysis],
        allow_synthetic=True,
    )

    assert rejected["claim_allowed"] is False
    assert rejected["analysis_audits"][0]["benchmark"] == "telecomts"
    assert rejected["analysis_audits"][0]["valid_evidence"] is True
    assert accepted["claim_allowed"] is True
    assert accepted["claim_tier"] == "last_resort_synthetic_only"
    assert accepted["selected_evidence"]["benchmark"] == "telecomts"


def test_claim_audit_rejects_telecomts_development_even_when_score_gate_passes(tmp_path):
    readiness = tmp_path / "readiness.json"
    analysis = tmp_path / "telecomts_development_analysis.json"
    readiness.write_text(json.dumps(_readiness(telecomts_ready=True)), encoding="utf-8")
    analysis.write_text(json.dumps(_analysis(
        benchmark="telecomts",
        baseline_selection={
            "method": "strongest_single_by_macro_then_micro_then_tokens",
            "resolved": "single_equal_calls",
        },
        passed=True,
        treatment="telecomts_shardrca_full",
        llm_mode=True,
        test_split=False,
        event_unit=True,
        frozen_prereg=False,
    )), encoding="utf-8")

    payload = audit_claim(
        readiness_path=readiness,
        analysis_paths=[analysis],
        allow_synthetic=True,
    )

    assert payload["claim_allowed"] is False
    audit = payload["analysis_audits"][0]
    assert audit["checks"]["gate_passed"] is True
    assert audit["checks"]["test_split"] is False
    assert audit["checks"]["frozen_prereg"] is False
    assert any("calibration evidence" in reason for reason in audit["reasons"])


def _readiness(
    *,
    openrca_ready: bool = False,
    telelogs_ready: bool = False,
    telelogs_agent_ready: bool = False,
    telecomts_ready: bool = False,
) -> dict:
    return {
        "headline_ready": openrca_ready,
        "fallback_ready": telelogs_ready or telelogs_agent_ready,
        "synthetic_ready": telecomts_ready,
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
            "telelogs": {
                "ready_for_headline": False,
                "ready_for_fallback": telelogs_ready,
                "reason": "" if telelogs_ready else "missing data",
            },
            "telelogs_agent": {
                "ready_for_headline": False,
                "ready_for_fallback": telelogs_agent_ready,
                "reason": "" if telelogs_agent_ready else "missing data",
            },
            "synthetic_telco": {
                "ready_for_synthetic_fallback": False,
            },
            "telecomts": {
                "ready_for_synthetic_fallback": telecomts_ready,
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
    official_tool_mode: bool | None = None,
    llm_mode: bool | None = None,
    test_split: bool | None = None,
    event_unit: bool | None = None,
    frozen_prereg: bool | None = None,
) -> dict:
    payload = {
        "paths": paths or ["results/openrca_paired_frozen.json"],
        "baseline": baseline_selection.get("resolved", "single_react_sc"),
        "baseline_selection": baseline_selection,
        "treatment": treatment,
        "clear_win_gate": {
            "passed": passed,
            "baseline": baseline_selection.get("resolved", "single_react_sc"),
            "treatment": treatment,
            "absolute_delta": 0.2,
            "observed_p": 0.01,
        },
    }
    if benchmark:
        payload["benchmark"] = benchmark
    if official_tool_mode is not None:
        payload["official_tool_mode"] = official_tool_mode
    if llm_mode is not None:
        payload["llm_mode"] = llm_mode
    if test_split is not None:
        payload["test_split"] = test_split
    if event_unit is not None:
        payload["event_unit"] = event_unit
    if frozen_prereg is not None:
        payload["frozen_prereg"] = frozen_prereg
    return payload

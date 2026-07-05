"""C: high-volume long-context split selection + a-priori power analysis."""
from __future__ import annotations

from telco_mas.shardrca.high_volume_split import (
    build_high_volume_split,
    build_prereg_draft,
    power_analysis,
)


def _hard_split_fixture():
    cases = []
    for i in range(20):
        vol = (i + 1) * 10_000_000  # 10MB .. 200MB
        cases.append({
            "runtime_case_id": f"RE2-TT-{i:04d}",
            "telemetry_bytes": vol,
            "telemetry_mb": vol / 1_000_000,
            "criteria": {"has_logs": i % 2 == 0, "has_traces": i % 3 != 0},
        })
    return {"meta": {"suite": "rcaeval_hard"}, "cases": cases}


def test_selects_top_volume_multimodal_and_excludes_locked():
    hard = _hard_split_fixture()
    locked = {"RE2-TT-0018", "RE2-TT-0019"}  # two of the highest-volume cases
    split = build_high_volume_split(hard, volume_quantile=0.75, exclude_case_ids=locked)
    ids = {c["runtime_case_id"] for c in split["cases"]}
    # Nothing from the locked holdout leaks in.
    assert ids.isdisjoint(locked)
    # Every selected case is above threshold and multi-modal.
    for c in split["cases"]:
        assert c["criteria"]["has_logs"] and c["criteria"]["has_traces"]
        assert c["telemetry_bytes"] >= split["meta"]["volume_threshold_bytes"]


def test_split_is_deterministic():
    hard = _hard_split_fixture()
    a = build_high_volume_split(hard)
    b = build_high_volume_split(hard)
    assert [c["runtime_case_id"] for c in a["cases"]] == [c["runtime_case_id"] for c in b["cases"]]


def test_power_analysis_reports_required_n_and_mde():
    pa = power_analysis(86, discordant_rate=0.35)
    # Small effects need far more pairs than a 51/86-case benchmark provides.
    assert pa["required_pairs_by_effect"]["delta_10pp"]["required_pairs"] > 86
    mde = pa["minimum_detectable_effect_at_n"]
    assert mde is not None and mde["delta"] > 0.10  # underpowered for a 10pp claim


def test_prereg_draft_is_draft_and_carries_power():
    hard = _hard_split_fixture()
    split = build_high_volume_split(hard)
    prereg = build_prereg_draft(split)
    assert prereg["status"] == "draft"
    assert prereg["treatment"] == "shardrca_full"
    assert "power_analysis" in prereg
    assert prereg["dataset"]["n"] == split["meta"]["n_selected"]
    # The freeze is explicitly gated on the weight fit + algorithm lock.
    assert any("frozen" in p for p in prereg["freeze_preconditions"])

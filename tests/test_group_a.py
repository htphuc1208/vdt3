from telco_mas.evaluation.run_group_a import analyze


def _row(case_id: str, system: str, hit: int, tokens: int) -> dict:
    return {
        "case_id": case_id,
        "system": system,
        "prior": "off",
        "hit_at_1": hit,
        "hit_at_3": hit,
        "mrr": float(hit),
        "total_tokens": tokens,
        "tool_calls": 1,
        "llm_calls": 1,
        "latency_s": 1.0,
    }


def test_analyze_supports_a_non_default_treatment():
    rows = [
        _row("a", "repaired_multi", 1, 20),
        _row("a", "single", 0, 10),
        _row("b", "repaired_multi", 1, 20),
        _row("b", "single", 1, 10),
    ]

    result = analyze(
        rows,
        ["repaired_multi", "single"],
        ["off"],
        treatment="repaired_multi",
    )["by_prior"]["off"]

    paired = result["pairwise_vs_treatment"]["single"]
    assert paired["treatment"] == "repaired_multi"
    assert paired["delta_hit_at_1_mean"] == 0.5
    assert paired["treatment_only_correct"] == 1
    assert paired["baseline_only_correct"] == 0
    assert result["treatment_token_ratio_over_baseline"]["single"] == 2.0

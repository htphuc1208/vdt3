import csv
import json

import pandas as pd

from telco_mas.evaluation.rcaeval_adapter import load_cases
from telco_mas.llm import LLMClient
from telco_mas.shardrca.catalog import build_catalog_for_case, extract_window_csv, make_component_group_shards
from telco_mas.shardrca.hard_split import build_hard_split
from telco_mas.shardrca.mining import (
    _Stats,
    _benjamini_hochberg,
    _log_count_shift_score,
    _welch_mean_shift,
    mine_metric_shifts,
    normalise_log_template,
)
from telco_mas.shardrca.miner import MinerWorker
from telco_mas.shardrca.panel import adjudicate_panel, metric_specialist
from telco_mas.shardrca.catalog import ShardSpec
from telco_mas.shardrca.board import Blackboard, CandidateEvidence, CandidateRootCause, Finding, WorkerDistribution
from telco_mas.shardrca.falsifier import falsify
from telco_mas.shardrca.fusion import fuse_candidate_evidence, fuse_worker_distributions
from telco_mas.shardrca.interaction import interact_candidate_evidence
from telco_mas.shardrca.result_analysis import analyze_results
from telco_mas.shardrca.runner import run_catalog, run_rcaeval_case
from telco_mas.shardrca.single_baseline import _candidate_from_agent, _single_react_user_prompt, run_single_react
from telco_mas.shardrca.synthesizer import SynthesizerResult
from telco_mas.shardrca.weights import FusionWeights


def test_windowed_extract_filters_csv_with_chunked_io(tmp_path):
    src = tmp_path / "metrics.csv"
    with src.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "svc_cpu"])
        writer.writeheader()
        for t in range(10):
            writer.writerow({"time": t, "svc_cpu": t * 10})

    out = tmp_path / "extract.csv"
    result = extract_window_csv(src, out, start_time=3, end_time=5, chunksize=2)

    frame = pd.read_csv(out)
    assert result["chunks"] == 5
    assert result["rows_written"] == 3
    assert frame["time"].tolist() == [3, 4, 5]


def test_metric_miner_ranks_injected_service_from_rcaeval_fixture(tmp_path):
    case = _fixture_case(tmp_path)
    catalog = build_catalog_for_case(case, compute_ranges=False)
    metric_paths = [item.path for item in catalog.files_by_modality("metrics")]

    findings = mine_metric_shifts(metric_paths, pivot_time=30, limit=5, chunksize=10)

    assert findings
    assert findings[0].component == "adservice"
    assert findings[0].signal == "cpu"
    assert findings[0].direction == "high"
    assert findings[0].metadata["score_method"] == "welch_t_bh_fdr_neg_log10_q"
    assert findings[0].metadata["multiple_testing"] == "benjamini_hochberg"
    assert findings[0].metadata["q_value"] <= findings[0].metadata["p_value"] * len(findings)


def test_log_template_normalizer_accepts_missing_scalar():
    assert normalise_log_template(float("nan")) == "nan"


def test_log_count_shift_score_is_volume_calibrated():
    assert _log_count_shift_score(100, 100) == 0.0
    assert 0 < _log_count_shift_score(0, 640) < 10
    assert _log_count_shift_score(0, 640) > _log_count_shift_score(0, 2)
    assert _log_count_shift_score(0, 2) == _log_count_shift_score(0, 2, support_tau=2.0)


def test_welch_metric_shift_uses_sample_size():
    small_pre = _stats_from_values([0.0, 1.0, 0.0, 1.0])
    small_post = _stats_from_values([1.0, 2.0, 1.0, 2.0])
    large_pre = _stats_from_values([0.0, 1.0] * 20)
    large_post = _stats_from_values([1.0, 2.0] * 20)

    small = _welch_mean_shift(small_pre, small_post)
    large = _welch_mean_shift(large_pre, large_post)

    assert large.t_stat > small.t_stat
    assert large.p_value < small.p_value


def test_benjamini_hochberg_adjusts_metric_family():
    q_values = _benjamini_hochberg([0.001, 0.02, 0.5])

    assert q_values == [0.003, 0.03, 0.5]


def test_component_score_caps_correlated_signals_per_modality():
    board = Blackboard(case_id="opaque")
    for index in range(10):
        board.add(Finding(shard_id="m0", modality="metrics", component="noisy", signal=f"copy-{index}", score=1.0))
    for index in range(3):
        board.add(Finding(shard_id="m1", modality="metrics", component="causal", signal=f"signal-{index}", score=2.0))

    assert board.component_score("noisy") == 3.0
    assert board.component_score("causal") == 6.0
    assert board.top_components(1)[0][0] == "causal"


def test_local_candidate_fusion_is_additive_not_convergence_boosted():
    board = Blackboard(case_id="opaque")
    board.extend([
        Finding(shard_id="m0", modality="metrics", component="single", signal="cpu", score=5.0, evidence_ptr="m0"),
        Finding(shard_id="m1", modality="metrics", component="converged", signal="cpu", score=3.0, evidence_ptr="m1"),
        Finding(shard_id="l1", modality="logs", component="converged", signal="error", score=3.0, evidence_ptr="l1"),
    ])
    evidence = [
        CandidateEvidence(component="single", reason_family="cpu", support_score=5.0, modality="metrics", shard_id="m0"),
        CandidateEvidence(component="converged", reason_family="cpu", support_score=3.0, modality="metrics", shard_id="m1"),
        CandidateEvidence(component="converged", reason_family="cpu", support_score=3.0, modality="logs", shard_id="l1"),
    ]

    result = fuse_candidate_evidence(evidence, board)
    legacy_boost_result = fuse_candidate_evidence(
        evidence,
        board,
        weights=FusionWeights(convergence_modality_bonus=99.0, convergence_shard_bonus=99.0),
    )

    assert result.winner.component == "converged"
    assert result.vote_breakdown["converged"] > result.vote_breakdown["single"]
    assert legacy_boost_result.vote_breakdown == result.vote_breakdown


def test_product_of_experts_fuses_independent_worker_distributions():
    board = Blackboard(case_id="opaque")
    board.add(Finding(
        shard_id="metrics",
        modality="metrics",
        component="docker_001",
        signal="cpu",
        score=4.0,
        evidence_ptr="metric#1",
    ))
    distributions = [
        WorkerDistribution(
            worker_id="metric",
            modality="metrics",
            candidate_scope=["docker_001", "docker_002"],
            candidates=[
                CandidateEvidence(
                    component="docker_001",
                    reason_family="CPU fault",
                    probability=0.8,
                    modality="metrics",
                    worker_id="metric",
                    evidence_ptrs=["metric#1"],
                )
            ],
            other_mass=0.2,
        ),
        WorkerDistribution(
            worker_id="trace",
            modality="traces",
            candidate_scope=["docker_001", "docker_002"],
            candidates=[
                CandidateEvidence(
                    component="docker_001",
                    reason_family="CPU fault",
                    probability=0.7,
                    modality="traces",
                    worker_id="trace",
                )
            ],
            other_mass=0.3,
        ),
    ]

    result = fuse_worker_distributions(
        distributions,
        board,
        components=["docker_001", "docker_002"],
        reasons=["CPU fault", "network delay"],
    )

    assert result.winner.component == "docker_001"
    assert result.winner.reason == "CPU fault"
    assert result.winner.evidence == ["metric#1"]


def test_worker_distribution_fusion_does_not_mutate_input_probabilities():
    board = Blackboard(case_id="opaque")
    candidates = [
        CandidateEvidence(
            component="docker_001",
            reason_family="CPU fault",
            probability=0.8,
            modality="metrics",
            worker_id="metric",
        ),
        CandidateEvidence(
            component="docker_002",
            reason_family="CPU fault",
            probability=0.8,
            modality="metrics",
            worker_id="metric",
        ),
    ]
    distributions = [
        WorkerDistribution(
            worker_id="metric",
            modality="metrics",
            candidate_scope=["docker_001", "docker_002"],
            candidates=candidates,
            other_mass=0.0,
        )
    ]

    fuse_worker_distributions(
        distributions,
        board,
        components=["docker_001", "docker_002"],
        reasons=["CPU fault"],
    )

    assert [item.probability for item in candidates] == [0.8, 0.8]


def test_autonomous_peer_interaction_adds_auditable_reviews():
    board = Blackboard(case_id="opaque")
    board.add(Finding(shard_id="metrics_a", modality="metrics", component="svc-a", signal="cpu", score=4.0, evidence_ptr="m#a"))
    board.add(Finding(shard_id="logs_b", modality="logs", component="svc-a", signal="error", score=3.0, evidence_ptr="l#a"))
    evidence = [
        CandidateEvidence(
            component="svc-a",
            reason_family="cpu",
            support_score=4.0,
            modality="metrics",
            worker_id="agent_metrics",
            shard_id="metrics_a",
            evidence_ptrs=["m#a"],
        ),
        CandidateEvidence(
            component="svc-a",
            reason_family="cpu",
            support_score=3.0,
            modality="logs",
            worker_id="agent_logs",
            shard_id="logs_b",
            evidence_ptrs=["l#a"],
        ),
    ]

    result = interact_candidate_evidence(evidence, board, llm=None)

    assert result.diagnostics["enabled"] is True
    assert len(result.evidence) > len(evidence)
    assert any(message["type"] == "peer_review" for message in result.transcript)
    assert any(item.worker_id.endswith("_peer_review") for item in result.evidence)


def test_deterministic_peer_support_uses_symmetric_evidence_budget():
    board = Blackboard(case_id="opaque")
    evidence = [
        CandidateEvidence(
            component="svc-a",
            reason_family="cpu",
            support_score=4.0,
            modality="metrics",
            worker_id="agent_metrics",
            shard_id="metrics",
            evidence_ptrs=["m#a"],
        ),
        CandidateEvidence(
            component="svc-a",
            reason_family="cpu",
            support_score=3.0,
            modality="logs",
            worker_id="agent_logs",
            shard_id="logs",
            evidence_ptrs=["l#a"],
        ),
    ]

    result = interact_candidate_evidence(evidence, board, llm=None)

    support_messages = [msg for msg in result.transcript if msg.get("verdict") == "support"]
    assert {msg["support"] for msg in support_messages} == {3.0, 4.0}
    assert all(msg["refute"] == 0.0 for msg in support_messages)
    assert result.diagnostics["review_policy"] == "symmetric_evidence_budget"


def test_disjoint_peer_scope_abstains_instead_of_inventing_counter_evidence():
    board = Blackboard(case_id="opaque")
    evidence = [
        CandidateEvidence(
            component="svc-a",
            reason_family="cpu",
            support_score=4.0,
            modality="metrics",
            worker_id="agent_metrics",
            shard_id="metrics",
            evidence_ptrs=["m#a"],
        ),
        CandidateEvidence(
            component="svc-b",
            reason_family="cpu",
            support_score=3.0,
            modality="logs",
            worker_id="agent_logs",
            shard_id="logs",
            evidence_ptrs=["l#b"],
        ),
    ]

    result = interact_candidate_evidence(evidence, board, llm=None)

    challenge = next(
        msg for msg in result.transcript
        if msg.get("from") == "agent_metrics" and msg.get("to") == "agent_logs"
    )
    assert challenge["verdict"] == "abstain"
    assert challenge["support"] == 0.0
    assert challenge["refute"] == 0.0
    refutations = [
        item for item in result.evidence
        if item.component == "svc-b" and item.worker_id == "agent_metrics_peer_review"
    ]
    assert refutations == []


def test_candidate_fusion_subtracts_refuting_peer_evidence():
    board = Blackboard(case_id="opaque")
    evidence = [
        CandidateEvidence(component="svc-a", reason_family="cpu", support_score=2.0, modality="metrics", shard_id="m"),
        CandidateEvidence(component="svc-a", reason_family="cpu", refute_score=5.0, modality="logs", shard_id="l"),
        CandidateEvidence(component="svc-b", reason_family="cpu", support_score=1.0, modality="traces", shard_id="t"),
    ]

    result = fuse_candidate_evidence(evidence, board)

    assert result.winner.component == "svc-b"
    assert result.vote_breakdown["svc-a"] < 0.0


def test_targeted_evidence_verifier_promotes_higher_board_score_without_magic_thresholds():
    board = Blackboard(case_id="opaque")
    board.add(Finding(shard_id="m", modality="metrics", component="svc-a", signal="cpu", score=1.0))
    board.add(Finding(shard_id="l", modality="logs", component="svc-b", signal="error", score=1.1))
    top = CandidateRootCause(component="svc-a", reason="cpu", confidence=0.9, score=10.0)
    runner = CandidateRootCause(component="svc-b", reason="log", confidence=0.1, score=1.0)

    result = falsify(
        board,
        [top, runner],
        top=top,
        min_evidence_score=999.0,
        runner_up_margin=999.0,
    )

    assert result.winner.component == "svc-b"
    assert result.falsified is True
    assert "decision_rule=promote_runner_if_board_score_strictly_higher" in result.checks
    assert "deprecated_threshold_args_ignored=true" in result.checks


def test_targeted_evidence_verifier_keeps_top_on_tie_or_weaker_runner():
    board = Blackboard(case_id="opaque")
    board.add(Finding(shard_id="m", modality="metrics", component="svc-a", signal="cpu", score=2.0))
    board.add(Finding(shard_id="l", modality="logs", component="svc-b", signal="error", score=2.0))
    top = CandidateRootCause(component="svc-a", reason="cpu", confidence=0.9, score=10.0)
    runner = CandidateRootCause(component="svc-b", reason="log", confidence=0.1, score=1.0)

    result = falsify(board, [top, runner], top=top)

    assert result.winner.component == "svc-a"
    assert result.falsified is False


def test_shardrca_rcaeval_runner_returns_label_safe_prediction(tmp_path):
    case = _fixture_case(tmp_path)

    pred = run_rcaeval_case(case, system="shardrca_full", llm=None, chunksize=10)

    assert pred.root == "adservice"
    assert pred.ranked_roots[0] == "adservice"
    assert pred.fault_type == "cpu"
    assert pred.accepted is True
    assert pred.total_tokens == 0
    assert pred.tool_calls >= 1


def test_full_uses_holistic_head_and_retains_mechanical_ablation(tmp_path):
    case = _fixture_case(tmp_path)
    catalog = build_catalog_for_case(case, compute_ranges=False)

    full = run_catalog(catalog, system="shardrca_full", llm=None, chunksize=10)
    mechanical = run_catalog(catalog, system="shardrca_mechanical", llm=None, chunksize=10)

    assert full.artifacts["decision_head"] == "grounded_multi_agent_panel"
    assert "grounded panel decision" in full.notes
    assert full.artifacts["architecture"] == "autonomous_peer_interaction_mas"
    assert mechanical.artifacts["decision_head"] == "mechanical_fusion"
    assert "mechanically fused" in mechanical.notes


def test_grounded_panel_can_select_a_specialist_candidate():
    board = Blackboard(case_id="opaque")
    board.add(Finding(shard_id="metrics", modality="metrics", component="svc-b", signal="cpu", score=4.0,
                      evidence_ptr="metrics#svc-b"))
    mechanical = SynthesizerResult(
        winner=CandidateRootCause(component="svc-a", reason="delay", confidence=0.6),
        candidates=[CandidateRootCause(component="svc-a", reason="delay", confidence=0.6)],
    )
    holistic = SynthesizerResult(
        winner=CandidateRootCause(component="svc-a", reason="delay", confidence=0.7),
        candidates=[CandidateRootCause(component="svc-a", reason="delay", confidence=0.7)],
    )
    llm = LLMClient(
        responder=lambda messages, tools: {
            "content": json.dumps({
                "root": "svc-b",
                "reason": "cpu",
                "confidence": 0.9,
                "ranked_roots": ["svc-b", "svc-a"],
                "rationale": "svc-b has the direct leading metric shift.",
            })
        },
        cache_enabled=False,
    )

    result, diagnostic = adjudicate_panel(
        mechanical,
        holistic,
        board,
        interaction_transcript=[],
        task_payload={"observability": {"top_metric_shifts": [
            {"service": "svc-b", "metric": "cpu", "score": 9.0},
            {"service": "svc-a", "metric": "latency", "score": 1.0},
        ]}},
        graph=None,
        llm=llm,
        k=1,
    )

    assert result.winner.component == "svc-b"
    assert result.winner.evidence == ["metrics#svc-b"]
    assert diagnostic["enabled"] is True
    assert diagnostic["valid_votes"] == 1


def test_metric_specialist_rejects_nonfinite_and_deduplicates_components():
    result = metric_specialist({"observability": {"top_metric_shifts": [
        {"service": "svc-a", "metric": "cpu", "score": float("nan")},
        {"service": "svc-b", "metric": "cpu", "score": 8.0},
        {"service": "svc-b", "metric": "mem", "score": 7.0},
        {"service": "svc-c", "metric": "cpu", "score": 2.0},
    ]}})

    assert result["proposal"] == "svc-b"
    assert [item["component"] for item in result["ranked_shifts"]] == ["svc-b", "svc-c"]
    assert result["confidence_margin"] == 0.8


def test_shardrca_local_fusion_runner_returns_prediction(tmp_path):
    case = _fixture_case(tmp_path, include_logs=True, include_traces=True)

    pred = run_rcaeval_case(case, system="shardrca_local_fusion", llm=None, chunksize=10)

    assert pred.system == "rcaeval_shardrca_local_fusion"
    assert pred.root == "adservice"
    assert pred.accepted is True
    assert pred.total_tokens == 0


def test_catalog_and_llm_prompts_do_not_expose_label_derived_path(tmp_path):
    case = _fixture_case(tmp_path)
    catalog = build_catalog_for_case(case, compute_ranges=False)
    board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())

    rendered = json.dumps(catalog.summary()) + board.compact_render() + _single_react_user_prompt(catalog)

    assert catalog.root not in rendered
    assert "adservice_cpu" not in rendered
    assert str(tmp_path) not in rendered


def test_single_sc_baseline_uses_same_mining_tools(tmp_path):
    case = _fixture_case(tmp_path)

    pred = run_rcaeval_case(case, system="single_sc", llm=None, chunksize=10)

    assert pred.root == "adservice"
    assert pred.ranked_roots[0] == "adservice"


def test_single_sc_uses_label_safe_component_universe(tmp_path):
    case = _fixture_case(tmp_path)
    metric_path = next((tmp_path / "RE1-OB" / "adservice_cpu" / "1").glob("*metric*.csv"))
    frame = pd.read_csv(metric_path)
    frame["ip-10-0-0-1_cpu"] = frame["adservice_cpu"] * 100
    frame.to_csv(metric_path, index=False)

    pred = run_rcaeval_case(case, system="single_sc", llm=None, chunksize=10)

    assert pred.root == "adservice"
    assert "ip-10-0-0-1" not in pred.ranked_roots


def test_single_react_uses_budgeted_local_tools(tmp_path):
    case = _fixture_case(tmp_path, include_logs=True, include_traces=True)
    catalog = build_catalog_for_case(case, compute_ranges=False)

    result = run_single_react(catalog, llm=None, max_tool_calls=2, chunksize=10)

    assert result.winner.component == "adservice"
    assert result.usage.tool_calls == 2
    assert {finding.modality for finding in result.board.findings} == {"metrics"}


def test_rcaeval_component_group_shards_are_disjoint(tmp_path):
    case = _fixture_case(tmp_path, include_logs=True, include_traces=True, extra_services=["cartservice", "checkoutservice"])
    catalog = build_catalog_for_case(case, compute_ranges=False)

    shards = make_component_group_shards(catalog, group_size=1, max_groups=3)
    metric_shards = [shard for shard in shards if shard.modality == "metrics"]

    assert len(metric_shards) >= 3
    assert all(len(shard.components) == 1 for shard in metric_shards[:3])
    assert len({tuple(shard.components) for shard in metric_shards[:3]}) == 3


def test_component_shards_share_one_physical_scan(monkeypatch, tmp_path):
    case = _fixture_case(tmp_path, extra_services=["cartservice"])
    catalog = build_catalog_for_case(case, compute_ranges=False)
    paths = [item.path for item in catalog.files_by_modality("metrics")]
    specs = [
        ShardSpec(
            shard_id=f"metrics_g{index}",
            modality="metrics",
            paths=paths,
            query_time=catalog.query_time,
            components=[component],
        )
        for index, component in enumerate(("adservice", "cartservice"))
    ]
    import telco_mas.shardrca.miner as miner_module

    original = miner_module.mine_shard
    calls = []

    def counted(spec, **kwargs):
        calls.append(spec)
        return original(spec, **kwargs)

    monkeypatch.setattr(miner_module, "mine_shard", counted)
    results = MinerWorker(limit=5, chunksize=10).run_many(specs)

    assert len(calls) == 1
    assert [result.shard_id for result in results] == ["metrics_g0", "metrics_g1"]
    assert {finding.component for finding in results[0].findings} == {"adservice"}
    assert {finding.shard_id for finding in results[0].findings} == {"metrics_g0"}


def test_single_react_runner_path_returns_prediction(tmp_path):
    case = _fixture_case(tmp_path)

    pred = run_rcaeval_case(case, system="single_react", llm=None, chunksize=10)

    assert pred.system == "rcaeval_single_react"
    assert pred.root == "adservice"
    assert pred.tool_calls >= 2


def test_code_retrieval_single_runner_path_returns_prediction(tmp_path):
    case = _fixture_case(tmp_path)

    pred = run_rcaeval_case(case, system="code_retrieval_single", llm=None, chunksize=10)

    assert pred.system == "rcaeval_code_retrieval_single"
    assert pred.root == "adservice"
    assert "code-retrieval" in pred.notes


def test_single_equal_tokens_uses_expanded_single_context_budget(tmp_path):
    case = _fixture_case(
        tmp_path,
        include_logs=True,
        include_traces=True,
        extra_services=[f"svc{i}" for i in range(12)],
    )

    budgeted = run_rcaeval_case(case, system="single_react", llm=None, chunksize=10)
    expanded = run_rcaeval_case(case, system="single_equal_tokens", llm=None, chunksize=10)

    assert expanded.system == "rcaeval_single_equal_tokens"
    assert expanded.tool_calls > budgeted.tool_calls
    assert "expanded tool budget" in expanded.notes


def test_single_react_accepts_list_shaped_agent_json(tmp_path):
    case = _fixture_case(tmp_path)
    catalog = build_catalog_for_case(case, compute_ranges=False)
    board = run_single_react(catalog, llm=None, max_tool_calls=2, chunksize=10).board

    candidate = _candidate_from_agent([
        {"component": "adservice", "reason": "cpu", "confidence": 0.8}
    ], board)

    assert candidate.component == "adservice"
    assert candidate.reason == "cpu"


def test_no_shard_ablation_runs_serial_path(tmp_path):
    case = _fixture_case(tmp_path)

    pred = run_rcaeval_case(case, system="no_shard", llm=None, chunksize=10)

    assert pred.root == "adservice"
    assert pred.notes.startswith("serial no-shard ablation")


def test_rcaeval_hard_split_is_label_safe(tmp_path):
    _fixture_case(tmp_path, dataset="RE2-TT", fault_label="adservice_cpu", include_logs=True, include_traces=True)

    split = build_hard_split(tmp_path, min_criteria=3)

    assert len(split["cases"]) == 1
    row = split["cases"][0]
    blob = json.dumps(row)
    case = load_cases(tmp_path)[0]
    assert row["runtime_case_id"] == case.runtime_case_id()
    assert "adservice" not in blob
    assert "cpu" not in blob
    assert "case_path" not in blob


def test_result_analysis_reports_pairwise_disagreements(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "summary": {},
        "rows": [
            {"case_id": "a", "system": "rcaeval_single_sc", "predicted_root": "svc-a",
             "true_root": "svc-a", "hit_at_1": True, "hit_at_3": True, "mrr": 1.0,
             "fault_accuracy": True, "total_tokens": 10, "llm_calls": 1, "tool_calls": 1, "latency_s": 1.0},
            {"case_id": "a", "system": "rcaeval_shardrca_full", "predicted_root": "svc-b",
             "true_root": "svc-a", "hit_at_1": False, "hit_at_3": False, "mrr": 0.0,
             "fault_accuracy": True, "total_tokens": 20, "llm_calls": 2, "tool_calls": 2, "latency_s": 2.0},
        ],
    }))

    analysis = analyze_results(path)

    assert analysis["benchmark"] == "rcaeval"
    assert analysis["disagreement_count"] == 1
    assert analysis["paired"]["hit_at_1"]["mcnemar"]["baseline_only_correct"] == 1
    assert analysis["usage"]["rcaeval_shardrca_full"]["total_tokens"] == 20


def test_result_analysis_resolves_strongest_single_baseline(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "summary": {},
        "rows": [
            _analysis_row("a", "rcaeval_single_react", False, 0.0, tokens=10),
            _analysis_row("a", "rcaeval_single_equal_tokens", True, 1.0, tokens=50),
            _analysis_row("a", "rcaeval_shardrca_full", True, 1.0, tokens=30),
            _analysis_row("b", "rcaeval_single_react", False, 0.0, tokens=10),
            _analysis_row("b", "rcaeval_single_equal_tokens", True, 1.0, tokens=50),
            _analysis_row("b", "rcaeval_shardrca_full", True, 1.0, tokens=30),
        ],
    }))

    analysis = analyze_results(path)

    assert analysis["baseline"] == "rcaeval_single_equal_tokens"
    assert analysis["baseline_selection"]["method"].startswith("strongest_single")
    assert analysis["clear_win_gate"]["absolute_delta"] == 0.0
    assert analysis["clear_win_gate"]["passed"] is False


def _analysis_row(case_id: str, system: str, hit: bool, mrr: float, *, tokens: int) -> dict:
    return {
        "case_id": case_id,
        "system": system,
        "predicted_root": "svc" if hit else "other",
        "true_root": "svc",
        "hit_at_1": hit,
        "hit_at_3": hit,
        "mrr": mrr,
        "fault_accuracy": hit,
        "total_tokens": tokens,
        "llm_calls": 1,
        "tool_calls": 1,
        "latency_s": 1.0,
    }


def _stats_from_values(values: list[float]) -> _Stats:
    stats = _Stats()
    stats.update(pd.Series(values))
    return stats


def _fixture_case(
    tmp_path,
    *,
    dataset="RE1-OB",
    fault_label="adservice_cpu",
    include_logs=False,
    include_traces=False,
    extra_services=None,
):
    case_dir = tmp_path / dataset / fault_label / "1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("30\n")
    rows = []
    extra_services = extra_services or []
    for t in range(60):
        row = {
            "time": t,
            "adservice_cpu": 90 if t >= 30 else 10,
            "cartservice_cpu": 12,
            "frontend_latency-50": 5,
        }
        for service in extra_services:
            row[f"{service}_cpu"] = 11
        rows.append(row)
    with (case_dir / "simple_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if include_logs:
        (case_dir / "logs.csv").write_text(
            "time,timestamp,container_name,message,level\n"
            "00:00,0,adservice,normal request,INFO\n"
            "00:01,31,adservice,error timeout stack trace,ERROR\n",
            encoding="utf-8",
        )
    if include_traces:
        (case_dir / "traces.csv").write_text(
            "time,traceID,spanID,serviceName,operationName,startTimeMillis,duration,statusCode\n"
            "0,t1,s1,adservice,GET,0,10,0\n"
            "31,t2,s2,adservice,GET,31,500,500\n",
            encoding="utf-8",
        )
    cases = load_cases(tmp_path)
    assert len(cases) == 1
    payload = cases[0].inference_payload()
    assert "ground_truth_root" not in payload
    assert "fault_type" not in payload
    return cases[0]

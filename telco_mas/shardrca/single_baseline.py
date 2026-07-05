"""Budget-aware single-agent baselines for ShardRCA comparisons."""
from __future__ import annotations

from dataclasses import dataclass

from ..llm import LLMClient
from ..schemas import UsageStats
from .board import Blackboard, CandidateRootCause
from .catalog import ShardSpec, TelemetryCatalog, candidate_components
from .local_tools import LOCAL_TOOL_SPECS, LocalToolSession, heuristic_probe
from .miner import MinerWorker
from .synthesizer import SynthesizerResult, fuse_candidates, heuristic_candidates, synthesize


@dataclass
class BaselineResult:
    board: Blackboard
    synthesis: SynthesizerResult
    usage: UsageStats

    @property
    def winner(self) -> CandidateRootCause:
        return self.synthesis.winner


def run_single_baseline(
    catalog: TelemetryCatalog,
    *,
    llm: LLMClient | None = None,
    self_consistency: bool = False,
    chunksize: int = 50_000,
    finding_limit: int = 30,
) -> BaselineResult:
    """Run one serial investigator over all available modality files."""

    board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
    usage = UsageStats()
    worker = MinerWorker(limit=finding_limit, chunksize=chunksize)
    component_filter = (
        candidate_components(catalog)
        if str(catalog.dataset).startswith("RE")
        else [str(item) for item in catalog.metadata.get("candidate_components", [])]
    )
    for modality in ("metrics", "logs", "traces"):
        files = catalog.files_by_modality(modality)
        if not files:
            continue
        result = worker.run(
            ShardSpec(
                shard_id=f"single_{modality}",
                modality=modality,
                paths=[item.path for item in files],
                query_time=catalog.query_time,
                start_time=catalog.time_range_start,
                end_time=catalog.time_range_end,
                components=component_filter,
                metadata={"single_agent": True, "file_count": len(files)},
            )
        )
        board.extend(result.findings)
        usage = usage.add(result.usage)
    synthesis = synthesize(board, llm=llm, k=3 if self_consistency else 1)
    usage = usage.add(synthesis.usage)
    return BaselineResult(board=board, synthesis=synthesis, usage=usage)


def run_single_react(
    catalog: TelemetryCatalog,
    *,
    llm: LLMClient | None = None,
    self_consistency: bool = False,
    chunksize: int = 50_000,
    finding_limit: int = 10,
    max_tool_calls: int = 3,
) -> BaselineResult:
    """Run a single-context ReAct investigator with budgeted local tools.

    Unlike ``run_single_baseline``, this does not receive a precomputed global
    blackboard. The LLM must decide which local shards to inspect, and the tool
    budget can expire before all modalities are queried.
    """

    if llm is None:
        session = LocalToolSession(
            catalog,
            chunksize=chunksize,
            finding_limit=finding_limit,
            max_tool_calls=max_tool_calls,
        )
        board = heuristic_probe(session)
        synthesis = synthesize(board, llm=None, k=3 if self_consistency else 1)
        return BaselineResult(
            board=board,
            synthesis=synthesis,
            usage=UsageStats(tool_calls=session.calls_used).add(synthesis.usage),
        )

    candidates = []
    usage = UsageStats()
    final_board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
    runs = 3 if self_consistency else 1
    for idx in range(runs):
        session = LocalToolSession(
            catalog,
            chunksize=chunksize,
            finding_limit=finding_limit,
            max_tool_calls=max_tool_calls,
        )
        agent_run = llm.run_agent(
            name=f"single_react_{idx}",
            system_prompt=SINGLE_REACT_SYSTEM,
            user_prompt=_single_react_user_prompt(catalog),
            tools_spec=LOCAL_TOOL_SPECS,
            dispatcher=session.dispatch,
            max_iters=max_tool_calls + 2,
        )
        usage = usage.add(agent_run.usage)
        board = session.board()
        final_board.extend(board.findings)
        candidate = _candidate_from_agent(agent_run.data, board)
        candidates.append(candidate)
    winner, vote_breakdown = fuse_candidates(candidates)
    synthesis = SynthesizerResult(winner=winner, candidates=candidates, vote_breakdown=vote_breakdown, usage=UsageStats())
    return BaselineResult(board=final_board, synthesis=synthesis, usage=usage)


SINGLE_REACT_SYSTEM = """You are a single-context RCA investigator.
You have budgeted local telemetry tools. You are NOT given the multi-agent blackboard.
First list shards, then inspect only the most useful shards within your tool budget.
Return ONLY JSON:
{"component": "service_or_component", "reason": "cpu|mem|disk|delay|loss|socket|network|db|unknown",
 "occurrence_time": "optional timestamp or null", "confidence": 0.0,
 "ranked_roots": ["service1", "service2", "service3"],
 "evidence": ["evidence_ptr", "..."], "rationale": "short evidence summary"}"""


def _single_react_user_prompt(catalog: TelemetryCatalog) -> str:
    import json

    payload = {
        "case_id": catalog.case_id,
        "catalog_summary": catalog.summary(),
        "instruction": "Identify the root-cause component and reason from label-safe telemetry.",
        "constraint": "Use only tool-returned evidence. Do not infer labels from IDs.",
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


def _candidate_from_agent(data: dict, board: Blackboard) -> CandidateRootCause:
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})
    if not isinstance(data, dict):
        data = {}
    component = str(data.get("component") or data.get("root") or "").strip()
    if not component:
        fallback = heuristic_candidates(board, k=1)[0]
        return fallback
    evidence = data.get("evidence") if isinstance(data.get("evidence"), list) else []
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    board_evidence = board.evidence_for(component, limit=8)
    return CandidateRootCause(
        component=component,
        reason=str(data.get("reason") or "unknown"),
        occurrence_time=str(data.get("occurrence_time")) if data.get("occurrence_time") else None,
        confidence=max(0.0, min(1.0, confidence)),
        rationale=str(data.get("rationale") or "single ReAct prediction from budgeted local tools"),
        evidence=[str(item) for item in evidence] or [item.evidence_ptr for item in board_evidence if item.evidence_ptr],
        score=board.component_score(component),
    )

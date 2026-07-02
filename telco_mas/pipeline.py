"""Top-level entry point: run one incident through the multi-agent system (or baseline)."""
from __future__ import annotations

from typing import Callable, Optional, Union

from .agents.orchestrator import MultiAgentOrchestrator, PipelineConfig
from .baseline import run_single_agent
from .environment.scenarios import (Scenario, build_incident, get_scenario, make_simulator)
from .knowledge.retriever import build_retriever
from .llm import LLMClient
from .schemas import Incident, PipelineResult
from .tools.registry import SessionContext

ProgressFn = Callable[[str], None]

# System label → ablation config for the multi-agent pipeline. "single" is the baseline.
MULTI_CONFIGS: dict[str, PipelineConfig] = {
    "multi": PipelineConfig(),
    "full": PipelineConfig(),
    "no_rag": PipelineConfig(use_rag=False),
    "no_consensus": PipelineConfig(use_consensus=False),
    "no_arbiter": PipelineConfig(use_arbiter=False),
}


def prepare(
    scenario: Union[str, Scenario],
    holdout: bool = False,
    kb_distractors: bool = False,
) -> tuple[SessionContext, Incident, Scenario]:
    """Build a fresh simulator + context + incident for a scenario.

    * ``holdout=True`` removes the exactly-matching SOP and historical incidents from
      the retriever, so the system cannot simply read the answer key (construct-validity
      control for the KB↔fault-library circularity).
    * ``kb_distractors=True`` adds plausible off-target SOPs/incidents so retrieval is
      non-trivial.
    """
    if isinstance(scenario, str):
        scenario = get_scenario(scenario)
    sim = make_simulator(scenario)
    exclude_sops = {scenario.remediation_sop} if holdout and scenario.remediation_sop else set()
    exclude_faults = {scenario.fault_type} if holdout else set()
    retriever = build_retriever(
        include_distractors=kb_distractors,
        exclude_sop_ids=exclude_sops,
        exclude_incident_fault_types=exclude_faults,
    )
    ctx = SessionContext.create(sim, retriever)
    incident = build_incident(scenario, sim)
    return ctx, incident, scenario


def run(
    scenario: Union[str, Scenario],
    mode: str = "multi",
    llm: Optional[LLMClient] = None,
    progress: Optional[ProgressFn] = None,
    holdout: bool = False,
    kb_distractors: bool = False,
) -> PipelineResult:
    """Run one scenario.

    mode='single' (baseline) or one of the multi-agent labels
    ('multi'/'full'/'no_rag'/'no_consensus'/'no_arbiter').
    """
    llm = llm or LLMClient()
    ctx, incident, _ = prepare(scenario, holdout=holdout, kb_distractors=kb_distractors)
    if mode == "single":
        return run_single_agent(incident, ctx, llm, progress=progress)
    config = MULTI_CONFIGS.get(mode, PipelineConfig())
    return MultiAgentOrchestrator(llm, ctx).run(incident, progress=progress, config=config)


def run_multi_agent(scenario: Union[str, Scenario], llm: Optional[LLMClient] = None, progress: Optional[ProgressFn] = None) -> PipelineResult:
    return run(scenario, mode="multi", llm=llm, progress=progress)

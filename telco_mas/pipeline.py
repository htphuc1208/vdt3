"""Top-level entry point: run one incident through the multi-agent system (or baseline)."""
from __future__ import annotations

from typing import Callable, Optional, Union

from .agents.orchestrator import MultiAgentOrchestrator
from .baseline import run_single_agent
from .environment.scenarios import (Scenario, build_incident, get_scenario, make_simulator)
from .knowledge.retriever import build_default_retriever
from .llm import LLMClient
from .schemas import Incident, PipelineResult
from .tools.registry import SessionContext

ProgressFn = Callable[[str], None]


def prepare(scenario: Union[str, Scenario]) -> tuple[SessionContext, Incident, Scenario]:
    """Build a fresh simulator + context + incident for a scenario."""
    if isinstance(scenario, str):
        scenario = get_scenario(scenario)
    sim = make_simulator(scenario)
    ctx = SessionContext.create(sim, build_default_retriever())
    incident = build_incident(scenario, sim)
    return ctx, incident, scenario


def run(
    scenario: Union[str, Scenario],
    mode: str = "multi",
    llm: Optional[LLMClient] = None,
    progress: Optional[ProgressFn] = None,
) -> PipelineResult:
    """Run one scenario. mode='multi' (agent team) or 'single' (baseline)."""
    llm = llm or LLMClient()
    ctx, incident, _ = prepare(scenario)
    if mode == "single":
        return run_single_agent(incident, ctx, llm, progress=progress)
    return MultiAgentOrchestrator(llm, ctx).run(incident, progress=progress)


def run_multi_agent(scenario: Union[str, Scenario], llm: Optional[LLMClient] = None, progress: Optional[ProgressFn] = None) -> PipelineResult:
    return run(scenario, mode="multi", llm=llm, progress=progress)

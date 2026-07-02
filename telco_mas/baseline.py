"""Single-agent baseline — the control condition for the benchmark.

One monolithic agent, all tools, no domain decomposition and no consensus. Produces a
``PipelineResult`` in the same shape as the multi-agent system so the two are directly
comparable.
"""
from __future__ import annotations

import time
from typing import Optional

from .agents.base import BaseAgent, as_float, as_str, incident_brief
from .llm import LLMClient
from .schemas import (ConsensusResult, Incident, PipelineResult, RemediationPlan,
                      UsageStats, ValidationResult)
from .tools.registry import SessionContext

SYSTEM = """You are an autonomous NOC engineer for a 5G network handling one incident end to end.
Investigate with the tools, determine the SINGLE root-cause element, then FIX it by calling
`apply_remediation` with a correct action on the correct element, and verify it recovered.

Remember: many alarms usually share one upstream root cause — follow the topology dependency chain.

When finished, respond with ONLY a JSON object:
{"root_cause": "one sentence",
 "faulty_element_id": "...",
 "fault_type": "...",
 "confidence": 0.0-1.0,
 "remediation_summary": "what you did to fix it",
 "resolved": true/false}"""


class SingleAgentBaseline(BaseAgent):
    name = "single_agent"
    tool_names = [
        "query_topology", "query_alarms", "query_kpis", "query_logs",
        "search_knowledge_base", "get_historical_incidents", "run_diagnostic", "apply_remediation",
    ]

    def run(self, incident: Incident, progress=None) -> PipelineResult:
        if progress:
            progress("Single monolithic agent handling the whole incident…")
        started = time.time()
        run = self.invoke(SYSTEM, incident_brief(incident))
        d = run.data
        consensus = ConsensusResult(
            root_cause=as_str(d.get("root_cause"), "Undetermined"),
            faulty_element_id=as_str(d.get("faulty_element_id")) or None,
            fault_type=as_str(d.get("fault_type")) or None,
            confidence=as_float(d.get("confidence"), 0.5),
            explanation="single-agent baseline (no expert team, no consensus vote)",
        )
        # Ground resolution in the actual network state, never just the model's claim.
        resolved = self.ctx.sim.is_healthy()
        result = PipelineResult(
            incident=incident,
            system="single_agent",
            consensus=consensus,
            remediation=RemediationPlan(summary=as_str(d.get("remediation_summary"))),
            validation=ValidationResult(resolved=resolved, notes=as_str(d.get("remediation_summary"))),
            trace=run.trace,
            usage=run.usage or UsageStats(),
            latency_s=round(time.time() - started, 2),
        )
        return result


def run_single_agent(incident: Incident, ctx: SessionContext, llm: Optional[LLMClient] = None, progress=None) -> PipelineResult:
    return SingleAgentBaseline(llm or LLMClient(), ctx).run(incident, progress=progress)

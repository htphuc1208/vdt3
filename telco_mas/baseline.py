"""Single-agent baseline — the control condition for the benchmark.

One monolithic agent, all tools, no domain decomposition and no consensus. Produces a
``PipelineResult`` in the same shape as the multi-agent system so the two are directly
comparable.
"""
from __future__ import annotations

import time
from typing import Optional

from .agents.base import BaseAgent, as_float, as_str, incident_brief
from .knowledge.fault_ontology import CANONICAL_FAULT_TYPES, canonicalize_fault_type
from .llm import LLMClient
from .schemas import (ConsensusResult, Incident, PipelineResult, RemediationPlan,
                      TraceStep, UsageStats, ValidationResult)
from .tools.registry import SessionContext

# STRONG baseline by design (biased AGAINST our own multi-agent hypothesis):
# full unrestricted tool access to every domain, a generous tool budget matching
# the whole team's, and explicit verify-and-retry instructions.
SINGLE_AGENT_MAX_ITERS = 24

SYSTEM = """You are an autonomous NOC engineer for a 5G network handling one incident end to end.
You have UNRESTRICTED access to every tool and every domain (RAN, transport, core, power), and a
generous tool budget — use it.

Investigate, determine the SINGLE root-cause element, then FIX it by calling `apply_remediation`
with a correct action on the correct element, and VERIFY recovery by re-checking alarms/KPIs.
If the fix did not take effect, re-investigate and try a different fix — you have budget for
multiple attempts.

Remember: many alarms usually share one upstream root cause — follow the topology dependency chain,
and run diagnostics on your suspect to confirm before fixing.
Report `fault_type` as a canonical ROOT-FAULT family, not as an alarm-condition name such as
CELL_DOWN, SYNC_HOLDOVER, or UP_LATENCY_DEGRADED. Allowed families:
__FAULT_TYPES__.

When finished, respond with ONLY a JSON object:
{"root_cause": "one sentence",
 "faulty_element_id": "...",
 "fault_type": "...",
 "confidence": 0.0-1.0,
 "remediation_summary": "what you did to fix it",
 "resolved": true/false}""".replace("__FAULT_TYPES__", ", ".join(CANONICAL_FAULT_TYPES))


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
        run = self.invoke(SYSTEM, incident_brief(incident), max_iters=SINGLE_AGENT_MAX_ITERS)
        d = run.data
        root_cause = as_str(d.get("root_cause"), "Undetermined")
        faulty_element_id = as_str(d.get("faulty_element_id")) or None
        fault_type = canonicalize_fault_type(
            as_str(d.get("fault_type")) or None,
            element=self.ctx.sim.topology.get(faulty_element_id) if faulty_element_id else None,
            alarms=incident.alarms,
            root_cause=root_cause,
        )
        consensus = ConsensusResult(
            root_cause=root_cause,
            faulty_element_id=faulty_element_id,
            fault_type=fault_type,
            confidence=as_float(d.get("confidence"), 0.5),
            explanation="single-agent baseline (no expert team, no consensus vote)",
        )
        action, target = _first_apply_remediation(run.trace)
        # Ground resolution in the actual network state, never just the model's claim.
        resolved = self.ctx.sim.is_healthy()
        result = PipelineResult(
            incident=incident,
            system="single_agent",
            consensus=consensus,
            remediation=RemediationPlan(summary=as_str(d.get("remediation_summary"))),
            remediation_action=action or as_str(d.get("remediation_summary")),
            remediation_target_element_id=target or consensus.faulty_element_id,
            validation=ValidationResult(resolved=resolved, notes=as_str(d.get("remediation_summary"))),
            trace=run.trace,
            usage=run.usage or UsageStats(),
            latency_s=round(time.time() - started, 2),
            remediation_attempts=_count_apply_remediation(run.trace),
        )
        return result


def _first_apply_remediation(trace: list[TraceStep]) -> tuple[str, str | None]:
    for step in trace:
        for call in step.tool_calls:
            if call.name == "apply_remediation":
                action = as_str(call.arguments.get("action"))
                target = as_str(call.arguments.get("element_id")) or None
                return action, target
    return "", None


def _count_apply_remediation(trace: list[TraceStep]) -> int:
    return sum(
        call.name == "apply_remediation"
        for step in trace
        for call in step.tool_calls
    )


def run_single_agent(incident: Incident, ctx: SessionContext, llm: Optional[LLMClient] = None, progress=None) -> PipelineResult:
    return SingleAgentBaseline(llm or LLMClient(), ctx).run(incident, progress=progress)

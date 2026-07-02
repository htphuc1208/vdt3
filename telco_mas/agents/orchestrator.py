"""Flow-of-Action orchestrator — the SOP-driven controller that runs the agent team.

It sequences the standard incident-handling procedure (detect → correlate → diagnose
in parallel → reach consensus → remediate → validate), threading state between agents
and accumulating a single trace + usage total for the whole run.

A ``PipelineConfig`` exposes real ablation switches so the contribution of each
component (RAG correlation, the consensus vote, the LLM arbiter) can be isolated:

* ``use_rag=False``     — skip the correlation agent and remove the knowledge-base
                          tools from the diagnosis experts (tests localisation without RAG).
* ``use_consensus=False`` — skip the consensus module; take the single most confident
                          expert (tests the fusion mechanism vs. best-expert).
* ``use_arbiter=False`` — run the numeric weighted vote but never call the LLM arbiter
                          (tests the arbiter's marginal value over the numeric vote).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from ..llm import LLMClient
from ..schemas import ConsensusResult, Incident, PipelineResult, UsageStats
from ..tools.registry import SessionContext
from .consensus import ConsensusModule
from .correlation import CorrelationAgent
from .detection import DetectionAgent
from .diagnosis import EXPERTS, DiagnosisAgent
from .remediation import RemediationAgent
from .validation import ValidationAgent

ProgressFn = Callable[[str], None]

_RAG_TOOLS = {"search_knowledge_base", "get_historical_incidents"}


@dataclass(frozen=True)
class PipelineConfig:
    """Ablation configuration for the multi-agent pipeline."""

    use_rag: bool = True
    use_consensus: bool = True
    use_arbiter: bool = True


FULL = PipelineConfig()


class MultiAgentOrchestrator:
    def __init__(self, llm: LLMClient, ctx: SessionContext) -> None:
        self.llm = llm
        self.ctx = ctx

    def run(
        self,
        incident: Incident,
        progress: Optional[ProgressFn] = None,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        config = config or FULL

        def note(msg: str) -> None:
            if progress:
                progress(msg)

        started = time.time()
        result = PipelineResult(incident=incident, system="multi_agent")
        usage = UsageStats()

        # 1. Triage ----------------------------------------------------------
        note("Detection/triage agent assessing the incident…")
        triage, trace, u = DetectionAgent(self.llm, self.ctx).run(incident)
        result.triage = triage
        result.trace += trace
        usage = usage.add(u)

        # 2. Knowledge correlation (ablatable) ------------------------------
        if config.use_rag:
            note("Correlation agent searching SOPs & historical incidents…")
            corr_notes, trace, u = CorrelationAgent(self.llm, self.ctx).run(incident, triage)
            result.correlation_notes = corr_notes
            result.trace += trace
            usage = usage.add(u)
        else:
            note("Correlation/RAG disabled (ablation).")
            corr_notes = ""

        expert_tools = None if config.use_rag else [
            t for t in DiagnosisAgent.tool_names if t not in _RAG_TOOLS
        ]

        # 3. Domain-expert diagnosis (parallel team) ------------------------
        for profile in EXPERTS:
            note(f"{profile.title} queued for parallel investigation…")
        with ThreadPoolExecutor(max_workers=len(EXPERTS)) as pool:
            futures = [
                pool.submit(
                    DiagnosisAgent(self.llm, self.ctx, profile).run,
                    incident, triage, corr_notes, expert_tools,
                )
                for profile in EXPERTS
            ]
        for future in futures:
            hyp, trace, u = future.result()
            result.hypotheses.append(hyp)
            result.trace += trace
            usage = usage.add(u)

        # 4. Consensus (ablatable) ------------------------------------------
        if config.use_consensus:
            note("Consensus module fusing expert hypotheses (weighted vote + arbiter)…")
            consensus, trace, u = ConsensusModule(self.llm, self.ctx).run(
                incident, result.hypotheses, use_arbiter=config.use_arbiter
            )
            result.trace += trace
            usage = usage.add(u)
        else:
            note("Consensus disabled (ablation) — taking the most confident expert.")
            consensus = self._best_expert_decision(result.hypotheses)
        result.consensus = consensus

        # 5. Remediation -----------------------------------------------------
        note("Remediation agent building an SOP-based plan…")
        plan, action, target, trace, u = RemediationAgent(self.llm, self.ctx).run(incident, consensus)
        result.remediation = plan
        result.trace += trace
        usage = usage.add(u)

        # 6. Validation (apply + verify) ------------------------------------
        note("Validation agent applying the fix and verifying recovery…")
        validation, trace, u = ValidationAgent(self.llm, self.ctx).run(plan, action, target)
        result.validation = validation
        result.trace += trace
        usage = usage.add(u)

        result.usage = usage
        result.latency_s = round(time.time() - started, 2)
        note("Done.")
        return result

    @staticmethod
    def _best_expert_decision(hypotheses: list) -> ConsensusResult:
        """no_consensus ablation: pick the single highest-confidence expert."""
        if not hypotheses:
            return ConsensusResult(root_cause="Undetermined", confidence=0.0,
                                   explanation="ablation: consensus disabled (no hypotheses)")
        top = max(hypotheses, key=lambda h: h.confidence)
        ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        return ConsensusResult(
            root_cause=top.root_cause,
            faulty_element_id=top.faulty_element_id,
            fault_type=top.fault_type,
            confidence=top.confidence,
            ranked=ranked,
            vote_breakdown={},
            explanation="ablation: consensus disabled — decision is the single most confident expert.",
        )

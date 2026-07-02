"""Flow-of-Action orchestrator — the SOP-driven controller that runs the agent team.

It sequences the standard incident-handling procedure (detect → correlate → diagnose
in parallel → reach consensus → remediate → validate), threading state between agents
and accumulating a single trace + usage total for the whole run.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..llm import LLMClient
from ..schemas import Incident, PipelineResult, UsageStats
from ..tools.registry import SessionContext
from .consensus import ConsensusModule
from .correlation import CorrelationAgent
from .detection import DetectionAgent
from .diagnosis import EXPERTS, DiagnosisAgent
from .remediation import RemediationAgent
from .validation import ValidationAgent

ProgressFn = Callable[[str], None]


class MultiAgentOrchestrator:
    def __init__(self, llm: LLMClient, ctx: SessionContext) -> None:
        self.llm = llm
        self.ctx = ctx

    def run(self, incident: Incident, progress: Optional[ProgressFn] = None) -> PipelineResult:
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

        # 2. Knowledge correlation ------------------------------------------
        note("Correlation agent searching SOPs & historical incidents…")
        corr_notes, trace, u = CorrelationAgent(self.llm, self.ctx).run(incident, triage)
        result.correlation_notes = corr_notes
        result.trace += trace
        usage = usage.add(u)

        # 3. Domain-expert diagnosis (the multi-agent team) -----------------
        for profile in EXPERTS:
            note(f"{profile.title} investigating…")
            hyp, trace, u = DiagnosisAgent(self.llm, self.ctx, profile).run(incident, triage, corr_notes)
            result.hypotheses.append(hyp)
            result.trace += trace
            usage = usage.add(u)

        # 4. Consensus -------------------------------------------------------
        note("Consensus module fusing expert hypotheses (weighted vote + arbiter)…")
        consensus, trace, u = ConsensusModule(self.llm, self.ctx).run(incident, result.hypotheses)
        result.consensus = consensus
        result.trace += trace
        usage = usage.add(u)

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

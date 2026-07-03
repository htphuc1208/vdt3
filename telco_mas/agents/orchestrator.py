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
from .consensus import ConsensusModule, verified_diagnostic_findings
from .correlation import CorrelationAgent
from .detection import DetectionAgent
from .diagnosis import EXPERTS, DiagnosisAgent
from .remediation import RemediationAgent
from .validation import ValidationAgent

ProgressFn = Callable[[str], None]

_RAG_TOOLS = {"search_knowledge_base", "get_historical_incidents"}

# Information partition (method v2): each expert can deep-inspect only its own
# domains — the mechanism that makes the committee's members genuinely diverse
# information sources instead of persona clones of the same model on the same data.
EXPERT_SCOPES: dict[str, frozenset[str]] = {
    "ran_expert": frozenset({"RAN"}),
    "transport_expert": frozenset({"TRANSPORT", "POWER"}),
    "core_expert": frozenset({"CORE"}),
}


@dataclass(frozen=True)
class PipelineConfig:
    """Ablation configuration for the multi-agent pipeline.

    * ``use_partition`` — scope each expert's deep tools to its own domain
      (information diversity by construction); off = all experts see everything.
    * ``use_debate``    — one cross-examination round when experts disagree on
      the root-cause element, before the vote.
    """

    use_rag: bool = True
    use_consensus: bool = True
    use_arbiter: bool = True
    use_partition: bool = True
    use_debate: bool = True


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

        # 3. Domain-expert diagnosis (parallel team, partitioned telemetry) --
        def expert_ctx(profile) -> SessionContext:
            if config.use_partition and profile.key in EXPERT_SCOPES:
                return self.ctx.scoped(EXPERT_SCOPES[profile.key])
            return self.ctx

        agents = {p.key: DiagnosisAgent(self.llm, expert_ctx(p), p) for p in EXPERTS}
        for profile in EXPERTS:
            note(f"{profile.title} queued for parallel investigation…")
        with ThreadPoolExecutor(max_workers=len(EXPERTS)) as pool:
            futures = {
                p.key: pool.submit(agents[p.key].run, incident, triage, corr_notes, expert_tools)
                for p in EXPERTS
            }
        expert_traces: dict[str, list] = {}
        hyps_by_expert: dict[str, object] = {}
        for key, future in futures.items():
            hyp, trace, u = future.result()
            hyps_by_expert[key] = hyp
            expert_traces[key] = trace
            result.trace += trace
            usage = usage.add(u)

        # 3b. Cross-examination round (ablatable): only when experts disagree.
        blamed = {h.faulty_element_id for h in hyps_by_expert.values() if h.faulty_element_id}
        if config.use_debate and len(blamed) > 1:
            note("Experts disagree — cross-examination round (each sees rivals' verified evidence)…")
            rivals_blocks = {
                key: "\n".join(
                    f"- {other.proposed_by}: element={other.faulty_element_id}, type={other.fault_type}, "
                    f"confidence={other.confidence:.2f} — {other.root_cause}\n"
                    f"    VERIFIED diagnostics: "
                    f"{'; '.join(verified_diagnostic_findings(expert_traces.get(other.proposed_by), other.faulty_element_id)) or 'NONE'}"
                    for okey, other in hyps_by_expert.items() if okey != key
                )
                for key in hyps_by_expert
            }
            with ThreadPoolExecutor(max_workers=len(EXPERTS)) as pool:
                rev_futures = {
                    key: pool.submit(agents[key].revise, incident, hyp, rivals_blocks[key], expert_tools)
                    for key, hyp in hyps_by_expert.items()
                }
            for key, future in rev_futures.items():
                hyp, trace, u = future.result()
                hyps_by_expert[key] = hyp
                expert_traces[key] = expert_traces.get(key, []) + trace
                result.trace += trace
                usage = usage.add(u)
            result.debate_rounds = 1

        result.hypotheses = list(hyps_by_expert.values())

        # 4. Consensus (ablatable) ------------------------------------------
        if config.use_consensus:
            note("Consensus module fusing expert hypotheses (verifiable-evidence vote)…")
            consensus, trace, u = ConsensusModule(self.llm, self.ctx).run(
                incident, result.hypotheses,
                expert_traces=expert_traces, use_arbiter=config.use_arbiter,
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
        result.remediation_action = action
        result.remediation_target_element_id = target
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

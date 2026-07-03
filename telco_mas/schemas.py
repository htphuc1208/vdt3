"""Typed data models shared across the whole system.

These pydantic/enum models are the contract between the simulator, the tools, the
agents and the evaluation harness.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Domain(str, Enum):
    RAN = "RAN"
    TRANSPORT = "TRANSPORT"
    CORE = "CORE"
    POWER = "POWER"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    WARNING = "WARNING"


class ElementType(str, Enum):
    CELL = "CELL"
    GNB = "GNB"
    AGG_SWITCH = "AGG_SWITCH"
    ROUTER = "ROUTER"
    FIBER_LINK = "FIBER_LINK"
    CORE_NF = "CORE_NF"
    POWER_UNIT = "POWER_UNIT"


# --------------------------------------------------------------------------- #
# Network model
# --------------------------------------------------------------------------- #
class NetworkElement(BaseModel):
    id: str
    name: str
    type: ElementType
    domain: Domain
    parent_id: Optional[str] = None
    site: Optional[str] = None


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
class Alarm(BaseModel):
    element_id: str
    severity: Severity
    name: str
    probable_cause: str


class KPISample(BaseModel):
    element_id: str
    metric: str
    value: float
    unit: str
    normal_range: tuple[float, float]
    is_anomalous: bool


class LogEntry(BaseModel):
    element_id: str
    level: str
    message: str


# --------------------------------------------------------------------------- #
# Incident (the input to the pipeline)
# --------------------------------------------------------------------------- #
class Incident(BaseModel):
    id: str
    title: str
    description: str
    alarms: list[Alarm] = Field(default_factory=list)
    affected_elements: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent outputs
# --------------------------------------------------------------------------- #
class TriageResult(BaseModel):
    severity: Severity
    suspected_domain: Domain
    affected_elements: list[str] = Field(default_factory=list)
    summary: str = ""


class Hypothesis(BaseModel):
    """A single candidate root cause produced by a diagnosis expert."""

    proposed_by: str
    root_cause: str
    faulty_element_id: Optional[str] = None
    fault_type: Optional[str] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    """Fused decision from the mABC-inspired weighted vote."""

    root_cause: str
    faulty_element_id: Optional[str] = None
    fault_type: Optional[str] = None
    confidence: float = 0.0
    ranked: list[Hypothesis] = Field(default_factory=list)
    vote_breakdown: dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class RemediationPlan(BaseModel):
    sop_id: Optional[str] = None
    summary: str = ""
    steps: list[str] = Field(default_factory=list)
    expected_outcome: str = ""


class ValidationResult(BaseModel):
    resolved: bool = False
    notes: str = ""
    recovered_kpis: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Trace + full pipeline result
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: str = ""


class TraceStep(BaseModel):
    agent: str
    role: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    def add(self, other: "UsageStats") -> "UsageStats":
        return UsageStats(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            tool_calls=self.tool_calls + other.tool_calls,
        )


class PipelineResult(BaseModel):
    incident: Incident
    triage: Optional[TriageResult] = None
    correlation_notes: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    consensus: Optional[ConsensusResult] = None
    remediation: Optional[RemediationPlan] = None
    remediation_action: str = ""
    remediation_target_element_id: Optional[str] = None
    validation: Optional[ValidationResult] = None
    trace: list[TraceStep] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)
    system: str = "multi_agent"
    latency_s: float = 0.0
    debate_rounds: int = 0
    # For multi-fault scenarios: whether the PRIMARY injected fault was cleared
    # (sim-grounded; set by the pipeline runner after execution).
    primary_fault_cleared: Optional[bool] = None

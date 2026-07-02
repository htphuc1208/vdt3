"""Common abstractions for external RCA benchmark suites."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExternalBenchmarkCase:
    """A label-safe benchmark case view used by adapters and runners."""

    case_id: str
    source: str
    instruction: str
    ground_truth_root: str
    fault_type: str = ""
    tags: list[str] = field(default_factory=list)
    observability: dict[str, Any] = field(default_factory=dict)
    label_extras: dict[str, Any] = field(default_factory=dict)

    def inference_payload(self) -> dict[str, Any]:
        """Payload safe to pass to a model. No ground-truth labels included."""
        return {
            "case_id": self.case_id,
            "source": self.source,
            "instruction": self.instruction,
            "tags": list(self.tags),
            "observability": self.observability,
        }


@dataclass
class ExternalPrediction:
    case_id: str
    system: str
    root: str
    ranked_roots: list[str] = field(default_factory=list)
    fault_type: str = ""
    accepted: bool = True
    confidence: float = 0.0
    latency_s: float = 0.0
    total_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    notes: str = ""

"""Label-safe support for telecom alarm-graph RCA benchmarks."""

from .dataset import TNRCACase, TNRCADataset, TNRCADatasetError
from .evaluator import evaluate_predictions
from .leakage import LeakageFinding, audit_graph_leakage, sanitize_runtime_graph
from .runner import TNRCARunResult, run_multi_agent, run_single_agent

__all__ = [
    "LeakageFinding",
    "TNRCACase",
    "TNRCADataset",
    "TNRCADatasetError",
    "TNRCARunResult",
    "audit_graph_leakage",
    "evaluate_predictions",
    "run_multi_agent",
    "run_single_agent",
    "sanitize_runtime_graph",
]

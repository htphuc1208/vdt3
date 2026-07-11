"""Audit and remove evaluator-only markers from telecom RCA graphs.

The public TeleCom-Bench example available in July 2026 contains a
``causedBy`` edge whose ``label`` is literally ``targetRootCause``.  That
field makes root-cause diagnosis trivial and is incompatible with a claim
that the ground truth is exclusive to ``label.json``.  The functions here
make that failure explicit and provide a deterministic clean-input protocol.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


_FORBIDDEN_VALUE_TOKENS = {
    "targetrootcause",
    "groundtruth",
    "groundtruthrootcause",
    "correctanswer",
    "goldanswer",
}
_FORBIDDEN_KEY_TOKENS = {
    "groundtruth",
    "groundtruthrootcause",
    "correctanswer",
    "goldanswer",
    "isrootcause",
    "istargetrootcause",
}


@dataclass(frozen=True)
class LeakageFinding:
    path: str
    kind: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "value": self.value}


def _token(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def audit_graph_leakage(graph: Any) -> list[LeakageFinding]:
    """Return explicit evaluator-marker leaks without treating candidates as leaks.

    Root-cause candidate names and IDs are legitimate runtime inputs, so their
    overlap with the label is intentionally not flagged.  Only discriminative
    answer markers are reported.
    """

    findings: list[LeakageFinding] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                key_token = _token(key)
                if key_token in _FORBIDDEN_KEY_TOKENS:
                    findings.append(LeakageFinding(child_path, "answer_key", str(child)))
                if not isinstance(child, (dict, list)) and _token(child) in _FORBIDDEN_VALUE_TOKENS:
                    findings.append(LeakageFinding(child_path, "answer_marker", str(child)))
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(graph, "")
    return findings


def sanitize_runtime_graph(graph: Any) -> Any:
    """Return a deep copy with explicit answer markers removed.

    Removing the entire marked field avoids leaving a binary presence/absence
    side channel.  Candidate root-cause nodes, topology, alarms, and ordinary
    relation types remain untouched.
    """

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, child in value.items():
                if _token(key) in _FORBIDDEN_KEY_TOKENS:
                    continue
                if not isinstance(child, (dict, list)) and _token(child) in _FORBIDDEN_VALUE_TOKENS:
                    continue
                result[key] = clean(child)
            return result
        if isinstance(value, list):
            return [clean(child) for child in value]
        return deepcopy(value)

    return clean(graph)

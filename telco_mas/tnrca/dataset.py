"""Dataset loader for TN-RCA530-style alarm knowledge graphs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .leakage import LeakageFinding, audit_graph_leakage, sanitize_runtime_graph


class TNRCADatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class TNRCACase:
    case_id: str
    graph: dict[str, Any]
    root_causes: tuple[str, ...]
    input_path: Path
    label_path: Path
    input_sha256: str
    leakage: tuple[LeakageFinding, ...]

    def runtime_graph(self, *, sanitize: bool = True) -> dict[str, Any]:
        return sanitize_runtime_graph(self.graph) if sanitize else self.graph


class TNRCADataset:
    """Load directory-per-case TN-RCA data or one public flat example.

    Supported layouts::

        root/test_000/input.json + label.json
        root/input.json + label.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise TNRCADatasetError(f"TN-RCA dataset directory does not exist: {self.root}")
        pairs = self._discover_pairs()
        if not pairs:
            raise TNRCADatasetError(
                f"No input.json/label.json case pairs found under {self.root}"
            )
        self.cases = tuple(self._load_case(case_id, input_path, label_path) for case_id, input_path, label_path in pairs)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[TNRCACase]:
        return iter(self.cases)

    def readiness(self, *, minimum_confirmatory_cases: int = 100) -> dict[str, Any]:
        leaking = [case for case in self.cases if case.leakage]
        return {
            "dataset": "TN-RCA-style telecom alarm graph RCA",
            "root": str(self.root),
            "case_count": len(self.cases),
            "minimum_confirmatory_cases": minimum_confirmatory_cases,
            "raw_cases_with_answer_markers": len(leaking),
            "raw_protocol_label_safe": not leaking,
            "clean_protocol_available": True,
            "sota_comparable": len(self.cases) >= minimum_confirmatory_cases and not leaking,
            "confirmatory_ready": len(self.cases) >= minimum_confirmatory_cases,
            "warnings": [
                *(
                    [
                        "Raw inputs contain explicit answer markers. Results on the sanitized graph "
                        "must be reported as a clean-input protocol and are not directly comparable "
                        "to scores produced from leaking raw inputs."
                    ]
                    if leaking
                    else []
                ),
                *(
                    [
                        f"Only {len(self.cases)} cases are available; at least "
                        f"{minimum_confirmatory_cases} are required for the planned held-out comparison."
                    ]
                    if len(self.cases) < minimum_confirmatory_cases
                    else []
                ),
            ],
            "leakage_examples": [
                {"case_id": case.case_id, **finding.to_dict()}
                for case in leaking[:5]
                for finding in case.leakage[:3]
            ],
        }

    def _discover_pairs(self) -> list[tuple[str, Path, Path]]:
        flat_input = self.root / "input.json"
        flat_label = self.root / "label.json"
        pairs: list[tuple[str, Path, Path]] = []
        if flat_input.is_file() and flat_label.is_file():
            pairs.append((self.root.name or "case_000", flat_input, flat_label))
        for input_path in sorted(self.root.rglob("input.json")):
            if input_path == flat_input:
                continue
            label_path = input_path.with_name("label.json")
            if label_path.is_file():
                pairs.append((str(input_path.parent.relative_to(self.root)), input_path, label_path))
        return pairs

    @staticmethod
    def _load_case(case_id: str, input_path: Path, label_path: Path) -> TNRCACase:
        try:
            input_bytes = input_path.read_bytes()
            graph = json.loads(input_bytes)
            label = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TNRCADatasetError(f"Invalid case {case_id}: {exc}") from exc
        if not isinstance(graph, dict) or not isinstance(label, dict):
            raise TNRCADatasetError(f"Case {case_id} input and label must be JSON objects")
        root_causes = _extract_root_causes(label)
        if not root_causes:
            raise TNRCADatasetError(f"Case {case_id} label contains no root-cause names")
        return TNRCACase(
            case_id=case_id,
            graph=graph,
            root_causes=tuple(sorted(root_causes)),
            input_path=input_path,
            label_path=label_path,
            input_sha256=hashlib.sha256(input_bytes).hexdigest(),
            leakage=tuple(audit_graph_leakage(graph)),
        )


def _extract_root_causes(label: dict[str, Any]) -> set[str]:
    causes: set[str] = set()
    for node in label.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_kind = str(node.get("label") or node.get("@class") or "").lower()
        labels = {str(value).lower() for value in node.get("labels", [])}
        if "rootcause" not in node_kind.replace("_", "") and "rootcause" not in labels:
            continue
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else node
        for key in ("causeName", "evalCause", "title", "cause_description"):
            value = str(properties.get(key) or "").strip()
            if value:
                causes.add(value)
                break
    return causes

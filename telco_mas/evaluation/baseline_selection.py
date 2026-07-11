"""Utilities for predeclared strongest-single baseline comparisons."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


STRONGEST_SINGLE_ALIASES = {"strongest_single", "best_single", "strongest-single", "best-single"}


def resolve_baseline(
    rows: list[dict[str, Any]],
    *,
    requested: str,
    treatment: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve a requested baseline, optionally selecting the strongest single baseline.

    Selection is deterministic and result-based over the frozen systems in the
    supplied result file: highest strict accuracy, then highest partial score,
    then lower token usage, then lexical system name. Candidates must be paired
    with the treatment on at least one case.
    """

    if requested not in STRONGEST_SINGLE_ALIASES:
        return requested, {
            "requested": requested,
            "resolved": requested,
            "method": "explicit",
        }

    candidates = []
    systems = sorted({str(row.get("system")) for row in rows})
    for system in systems:
        if system == treatment or not is_single_system(system):
            continue
        system_rows = [row for row in rows if str(row.get("system")) == system]
        paired_cases = _paired_cases(rows, system, treatment)
        if not paired_cases:
            continue
        strict_values = [1.0 if row.get("strict_correct") else 0.0 for row in system_rows]
        score_values = [float(row.get("score") or 0.0) for row in system_rows]
        total_tokens = sum(int(row.get("total_tokens") or 0) for row in system_rows)
        candidates.append({
            "system": system,
            "strict_accuracy": mean(strict_values) if strict_values else 0.0,
            "score": mean(score_values) if score_values else 0.0,
            "total_tokens": total_tokens,
            "paired_cases": len(paired_cases),
        })

    if not candidates:
        raise ValueError(
            f"No single-agent baseline systems are paired with treatment '{treatment}'. "
            "Pass an explicit --baseline or include a single* / *single system."
        )

    selected = sorted(
        candidates,
        key=lambda item: (
            -float(item["strict_accuracy"]),
            -float(item["score"]),
            int(item["total_tokens"]),
            str(item["system"]),
        ),
    )[0]
    return str(selected["system"]), {
        "requested": requested,
        "resolved": selected["system"],
        "method": "strongest_single_by_strict_then_score_then_tokens",
        "candidates": candidates,
    }


def is_single_system(system: str) -> bool:
    lower = system.lower()
    if lower.startswith("single") or lower.startswith("rcaeval_single"):
        return True
    return lower in {"code_retrieval_single", "instruction_llm"}


def _paired_cases(rows: list[dict[str, Any]], baseline: str, treatment: str) -> set[str]:
    by_case: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        case_id = str(row.get("case_id") or row.get("row_id") or row.get("scenario") or row.get("id"))
        by_case[case_id].add(str(row.get("system")))
    return {
        case_id
        for case_id, systems in by_case.items()
        if baseline in systems and treatment in systems
    }

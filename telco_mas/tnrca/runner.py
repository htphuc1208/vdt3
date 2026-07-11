"""Call-matched single and multi-agent runners for telecom alarm graphs."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .dataset import TNRCACase


@dataclass(frozen=True)
class TNRCARunResult:
    case_id: str
    system: str
    root_causes: tuple[str, ...]
    usage: UsageStats
    artifacts: dict[str, Any]


_OUTPUT_SCHEMA = (
    'Return ONLY JSON: {"root_causes":[{"cause_description":"exact candidate name",'
    '"evidence":["node/edge IDs"]}],"rationale":"brief grounded explanation"}.'
)


def run_single_agent(
    case: TNRCACase,
    llm: LLMClient,
    *,
    calls: int = 4,
) -> TNRCARunResult:
    """Run self-consistent serial readers over the complete sanitized graph."""

    if calls < 1:
        raise ValueError("calls must be positive")
    graph = case.runtime_graph(sanitize=True)
    universe = _candidate_universe(graph)
    prompt = _case_prompt(graph, universe)
    opinions: list[dict[str, Any]] = []
    usage = UsageStats()
    for index in range(calls):
        response = llm.chat(
            [
                {"role": "system", "content": _single_system(index)},
                {"role": "user", "content": prompt},
            ],
            force_json=True,
        )
        usage = usage.add(response.usage)
        parsed = extract_json(response.content)
        roots = _parse_roots(parsed, universe)
        opinions.append({"run": index, "root_causes": roots, "raw": parsed})
    selected, votes = _majority_roots([opinion["root_causes"] for opinion in opinions], calls)
    return TNRCARunResult(
        case_id=case.case_id,
        system="single_self_consistency",
        root_causes=tuple(selected),
        usage=usage,
        artifacts={"call_budget": calls, "opinions": opinions, "votes": votes, "candidate_universe": universe},
    )


def run_multi_agent(case: TNRCACase, llm: LLMClient) -> TNRCARunResult:
    """Run three evidence-specialists and one constrained adjudicator.

    This uses four LLM calls, matching the default single-agent self-consistency
    baseline. Specialists receive different graph views; the adjudicator receives
    their grounded proposals rather than another copy of the full graph.
    """

    graph = case.runtime_graph(sanitize=True)
    universe = _candidate_universe(graph)
    views = _graph_views(graph)
    opinions: list[dict[str, Any]] = []
    usage = UsageStats()
    for role in ("alarm_semantics", "resource_topology", "causal_skeptic"):
        response = llm.chat(
            [
                {"role": "system", "content": _specialist_system(role)},
                {"role": "user", "content": _case_prompt(views[role], universe)},
            ],
            force_json=True,
        )
        usage = usage.add(response.usage)
        parsed = extract_json(response.content)
        opinions.append({"role": role, "root_causes": _parse_roots(parsed, universe), "raw": parsed})

    eligible = [
        candidate
        for candidate in universe
        if any(candidate in opinion["root_causes"] for opinion in opinions)
    ]
    adjudication_payload = {
        "candidate_universe": universe,
        "eligible_candidates": eligible,
        "specialist_opinions": opinions,
        "instruction": (
            "Select all and only root causes supported by the specialist evidence. "
            "You may choose only eligible_candidates; abstain with an empty list if none are grounded."
        ),
    }
    response = llm.chat(
        [
            {"role": "system", "content": _adjudicator_system()},
            {"role": "user", "content": json.dumps(adjudication_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        force_json=True,
    )
    usage = usage.add(response.usage)
    adjudication = extract_json(response.content)
    selected = _parse_roots(adjudication, eligible)
    return TNRCARunResult(
        case_id=case.case_id,
        system="multi_specialist_adjudicator",
        root_causes=tuple(selected),
        usage=usage,
        artifacts={
            "call_budget": 4,
            "specialists": opinions,
            "eligible_candidates": eligible,
            "adjudication": adjudication,
            "candidate_universe": universe,
        },
    )


def _single_system(index: int) -> str:
    lenses = ("causal paths", "alarm semantics", "equipment topology", "counterfactual falsification")
    return (
        "You are one serial telecom RAN alarm-graph RCA agent. Analyze the complete graph independently. "
        f"Use {lenses[index % len(lenses)]} as a checking lens, but integrate all available evidence. "
        "Never infer an answer from case IDs or absent labels. " + _OUTPUT_SCHEMA
    )


def _specialist_system(role: str) -> str:
    instructions = {
        "alarm_semantics": (
            "You are the alarm-semantics specialist. Compare target and propagated alarm titles, codes, times, "
            "and candidate descriptions. Distinguish a triggering fault from downstream alarms."
        ),
        "resource_topology": (
            "You are the RAN resource-topology specialist. Trace BBU/RRU/board/port/cable dependencies and "
            "prefer a cause whose equipment position explains the observed alarm cascade."
        ),
        "causal_skeptic": (
            "You are the causal-path skeptic. Enumerate legal graph paths, challenge ambiguous candidates, and "
            "reject candidates supported only by an undirected proximity or a loud symptom."
        ),
    }
    return instructions[role] + " Use only the supplied sanitized graph. " + _OUTPUT_SCHEMA


def _adjudicator_system() -> str:
    return (
        "You are the final telecom RCA adjudicator. Reconcile independent specialists, favor direct path evidence, "
        "and do not turn agreement into proof. You may select only eligible candidates. " + _OUTPUT_SCHEMA
    )


def _case_prompt(graph: dict[str, Any], universe: list[str]) -> str:
    payload = {
        "candidate_universe": universe,
        "graph": graph,
        "constraint": "The graph has been stripped of evaluator answer markers. Use only graph evidence.",
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _candidate_universe(graph: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        kind = str(node.get("@class") or node.get("label") or "").replace("_", "").casefold()
        labels = {str(value).replace("_", "").casefold() for value in node.get("labels", [])}
        if "rootcause" not in kind and "alarmcause" not in kind and not ({"rootcause", "alarmcause"} & labels):
            continue
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else node
        name = next(
            (str(properties.get(key) or "").strip() for key in ("title", "causeName", "evalCause", "cause_description") if str(properties.get(key) or "").strip()),
            "",
        )
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _graph_views(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]

    def node_kind(node: dict[str, Any]) -> str:
        return str(node.get("@class") or node.get("label") or "").replace("_", "").casefold()

    def node_id(node: dict[str, Any]) -> str:
        return str(node.get("@rid") or node.get("id") or node.get("_id") or "")

    candidate_nodes = [node for node in nodes if "rootcause" in node_kind(node) or "alarmcause" in node_kind(node)]
    alarm_nodes = [node for node in nodes if "alarm" in node_kind(node)]
    resource_nodes = [node for node in nodes if node not in candidate_nodes and node not in alarm_nodes]

    def induced(selected: list[dict[str, Any]], *, include_causal: bool = False) -> dict[str, Any]:
        selected_ids = {node_id(node) for node in selected if node_id(node)}
        selected_edges = []
        for edge in edges:
            endpoints = {str(edge.get("in") or ""), str(edge.get("out") or "")}
            relation = str(edge.get("@class") or edge.get("type") or "").casefold()
            if endpoints <= selected_ids or (include_causal and "caus" in relation and endpoints & selected_ids):
                selected_edges.append(edge)
        return {"nodes": selected, "edges": selected_edges}

    return {
        "alarm_semantics": induced([*alarm_nodes, *candidate_nodes], include_causal=True),
        "resource_topology": induced([*resource_nodes, *candidate_nodes, *alarm_nodes]),
        "causal_skeptic": graph,
    }


def _parse_roots(data: dict[str, Any], universe: list[str]) -> list[str]:
    if not isinstance(data, dict) or not universe:
        return []
    values: list[str] = []
    roots = data.get("root_causes")
    if isinstance(roots, list):
        for item in roots:
            if isinstance(item, dict):
                values.append(str(item.get("cause_description") or item.get("title") or item.get("root") or ""))
            else:
                values.append(str(item))
    elif isinstance(roots, str):
        values.append(roots)
    for key in ("root", "cause_description"):
        if isinstance(data.get(key), str):
            values.append(data[key])
    allowed = {candidate.casefold().strip(): candidate for candidate in universe}
    selected: list[str] = []
    for value in values:
        canonical = allowed.get(value.casefold().strip())
        if canonical and canonical not in selected:
            selected.append(canonical)
    return selected


def _majority_roots(opinions: list[list[str]], calls: int) -> tuple[list[str], dict[str, int]]:
    counts = Counter(candidate for roots in opinions for candidate in set(roots))
    threshold = math.floor(calls / 2) + 1
    selected = [candidate for candidate, count in counts.most_common() if count >= threshold]
    if not selected and counts:
        best_count = counts.most_common(1)[0][1]
        selected = [candidate for candidate, count in counts.most_common() if count == best_count]
    return selected, dict(counts)

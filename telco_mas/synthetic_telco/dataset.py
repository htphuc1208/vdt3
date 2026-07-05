"""Export and validate a telecom-valid synthetic RCA fallback dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..environment.scenarios import Scenario, build_incident, get_scenario, list_scenario_ids, make_simulator
from ..schemas import Domain


DEFAULT_SUITE = "telco_v3"
FORBIDDEN_RUNTIME_KEYS = {
    "source_scenario_id",
    "root_element_id",
    "acceptable_elements",
    "fault_type",
    "root_cause_keywords",
    "remediation_sop",
    "remediation_keywords",
    "causal_graph",
}


def build_dataset(
    *,
    suite: str = DEFAULT_SUITE,
    seed: int | None = None,
    seed_source: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable synthetic telecom RCA dataset.

    The runtime half is observable incident/telemetry/topology data. The labels
    half is evaluator-only. This is deliberately an export artifact, not a new
    benchmark result.
    """

    design: dict[str, Any] | None = None
    if suite == "telco_v4":
        if seed is None or not seed_source:
            raise ValueError(
                "telco_v4 requires an explicit --seed and --seed-source chosen before outcome inspection"
            )
        from .holdout import build_holdout_scenarios, design_manifest

        scenarios = build_holdout_scenarios(seed)
        design = design_manifest(seed, scenarios, seed_source=seed_source)
    else:
        scenarios = [get_scenario(sid) for sid in list_scenario_ids(suite)]
    cases = [_case_payload(scenario) for scenario in scenarios]
    validation = validate_dataset({"cases": cases})
    role = (
        "confirmatory synthetic holdout; generate once after algorithm freeze and never tune on outcomes"
        if suite == "telco_v4"
        else "synthetic fallback only; real OpenRCA/TN-RCA evidence remains preferred"
    )
    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "name": f"synthetic_{suite}_telecom_rca",
            "suite": suite,
            "role_in_claim": role,
            "scientific_basis": [
                "Topology-dependent alarm propagation across RAN/transport/core/power layers",
                "Alarm flood plus KPI/log evidence windows",
                "Evaluator-only root/fault/remediation labels",
                "Difficulty bins from topology size, alarm fanout, distractors, missing telemetry, and multi-fault tags",
                "3GPP TS 28.111 alarm identity, severity, probable-cause, and lifecycle concepts",
            ],
            "case_count": len(cases),
            "difficulty_counts": dict(Counter(case["labels"]["difficulty"] for case in cases)),
            "content_sha256": _canonical_sha256(cases),
            "design": design,
            "validation": validation,
        },
        "cases": cases,
    }


def validate_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    cases = dataset.get("cases", [])
    errors: list[str] = []
    runtime_ids = set()
    for index, case in enumerate(cases):
        runtime = case.get("runtime", {})
        labels = case.get("labels", {})
        runtime_id = runtime.get("runtime_case_id")
        if not runtime_id:
            errors.append(f"case[{index}] missing runtime_case_id")
        elif runtime_id in runtime_ids:
            errors.append(f"duplicate runtime_case_id {runtime_id}")
        runtime_ids.add(runtime_id)
        leaked = sorted(_nested_keys(runtime) & FORBIDDEN_RUNTIME_KEYS)
        if leaked:
            errors.append(
                f"case[{index}] runtime contains evaluator-only label key(s): {', '.join(leaked)}"
            )
        if not runtime.get("incident", {}).get("alarms"):
            errors.append(f"case[{index}] has no runtime alarms")
        if not runtime.get("telemetry", {}).get("kpis"):
            errors.append(f"case[{index}] has no runtime KPI samples")
        if not labels.get("root_element_id") or not labels.get("fault_type"):
            errors.append(f"case[{index}] missing evaluator labels")
        if not labels.get("causal_graph", {}).get("edges"):
            errors.append(f"case[{index}] missing causal graph edges")
        difficulty = labels.get("difficulty")
        if difficulty not in {"easy", "middle", "hard"}:
            errors.append(f"case[{index}] invalid difficulty {difficulty!r}")
        if runtime.get("suite") == "telco_v4":
            for alarm in runtime.get("incident", {}).get("alarms", []):
                if not alarm.get("raised_at"):
                    errors.append(f"case[{index}] telco_v4 alarm missing raised_at")
                    break
            for sample in runtime.get("telemetry", {}).get("kpis", []):
                if not sample.get("timestamp"):
                    errors.append(f"case[{index}] telco_v4 KPI missing timestamp")
                    break
            for entry in runtime.get("telemetry", {}).get("logs", []):
                if not entry.get("timestamp"):
                    errors.append(f"case[{index}] telco_v4 log missing timestamp")
                    break
    return {
        "ok": not errors,
        "case_count": len(cases),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a telecom-valid synthetic RCA fallback dataset.")
    parser.add_argument("--suite", default=DEFAULT_SUITE, choices=["telco_v1", "telco_v2", "telco_v3", "telco_v4", "all"])
    parser.add_argument("--seed", type=int, default=None,
                        help="required generation seed for telco_v4; choose and record before evaluation")
    parser.add_argument("--seed-source", default=None,
                        help="auditable v4 seed provenance, e.g. a predeclared NIST beacon pulse URL")
    parser.add_argument("--out", default="results/synthetic_telco_rca_dataset.json")
    args = parser.parse_args(argv)

    try:
        payload = build_dataset(suite=args.suite, seed=args.seed, seed_source=args.seed_source)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "case_count": payload["meta"]["case_count"],
        "difficulty_counts": payload["meta"]["difficulty_counts"],
        "validation_ok": payload["meta"]["validation"]["ok"],
    }, indent=2))
    return 0 if payload["meta"]["validation"]["ok"] else 2


def _case_payload(scenario: Scenario) -> dict[str, Any]:
    sim = make_simulator(scenario)
    incident = build_incident(scenario, sim)
    runtime_id = _runtime_id(scenario)
    kpis = [sample.model_dump(mode="json") for sample in sim.get_kpis()]
    alarms = [alarm.model_dump(mode="json") for alarm in incident.alarms]
    logs = [entry.model_dump(mode="json") for entry in sim.get_logs(limit=80)]
    topology = _topology_payload(sim, alarms)
    affected = sorted({alarm["element_id"] for alarm in alarms} | {sample["element_id"] for sample in kpis if sample["is_anomalous"]})
    return {
        "runtime": {
            "runtime_case_id": runtime_id,
            "suite": scenario.suite,
            "incident": {
                "title": incident.title,
                "description": incident.description,
                "alarms": alarms,
                "affected_elements": affected,
            },
            "telemetry": {
                "kpis": kpis,
                "logs": logs,
            },
            "topology": topology,
            "instructions": "Identify the root-cause element, fault family, and remediation from observable telemetry only.",
        },
        "labels": {
            "source_scenario_id": scenario.id,
            "root_element_id": scenario.element_id,
            "acceptable_elements": list(scenario.acceptable_elements),
            "fault_type": scenario.fault_type,
            "domain": scenario.domain.value,
            "root_cause_keywords": scenario.root_cause_keywords,
            "remediation_sop": scenario.remediation_sop,
            "remediation_keywords": scenario.remediation_keywords,
            "stress_tags": list(scenario.stress_tags),
            "secondary_faults": [list(item) for item in scenario.secondary_faults],
            "causal_graph": _causal_graph(sim, scenario),
            "difficulty": _difficulty(scenario, len(alarms), len(affected), len(sim.topology.all())),
        },
    }


def _runtime_id(scenario: Scenario) -> str:
    digest = hashlib.sha256(f"{scenario.suite}:{scenario.id}".encode("utf-8")).hexdigest()[:12]
    return f"ST-{scenario.suite}-{digest}"


def load_scenarios(dataset: dict[str, Any], *, verify_runtime: bool = True) -> list[Scenario]:
    """Rebuild simulator scenarios from a frozen artifact and verify its runtime view."""
    validation = validate_dataset(dataset)
    if not validation["ok"]:
        raise ValueError("synthetic dataset validation failed: " + "; ".join(validation["errors"]))
    committed_hash = dataset.get("meta", {}).get("content_sha256")
    observed_hash = _canonical_sha256(dataset.get("cases", []))
    if committed_hash and committed_hash != observed_hash:
        raise ValueError(
            "dataset content hash mismatch; the frozen runtime or evaluator labels were modified"
        )
    scenarios: list[Scenario] = []
    for case in dataset.get("cases", []):
        runtime = case["runtime"]
        labels = case["labels"]
        scenario = Scenario(
            id=labels["source_scenario_id"],
            title=runtime["incident"]["title"],
            description=runtime["incident"]["description"],
            fault_type=labels["fault_type"],
            element_id=labels["root_element_id"],
            domain=Domain(labels["domain"]),
            root_cause_keywords=list(labels.get("root_cause_keywords", [])),
            remediation_sop=labels.get("remediation_sop", ""),
            remediation_keywords=list(labels.get("remediation_keywords", [])),
            suite=runtime["suite"],
            stress_tags=tuple(labels.get("stress_tags", [])),
            secondary_faults=tuple(tuple(item) for item in labels.get("secondary_faults", [])),
            topology="large" if runtime["suite"] in {"telco_v3", "telco_v4"} else "small",
        )
        if verify_runtime and _case_payload(scenario)["runtime"] != runtime:
            raise ValueError(
                f"runtime reconstruction mismatch for {runtime['runtime_case_id']}; "
                "the frozen artifact and current simulator code differ"
            )
        scenarios.append(scenario)
    return scenarios


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_nested_keys(child))
    return keys


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _topology_payload(sim, alarms: list[dict[str, Any]]) -> dict[str, Any]:
    observed_ids = sorted({alarm["element_id"] for alarm in alarms})
    expanded = set(observed_ids)
    for element_id in observed_ids:
        element = sim.topology.get(element_id)
        if element and element.parent_id:
            expanded.add(element.parent_id)
        for child in sim.topology.children(element_id)[:8]:
            expanded.add(child.id)
    nodes = []
    edges = []
    for element_id in sorted(expanded):
        element = sim.topology.get(element_id)
        if not element:
            continue
        nodes.append({
            "id": element.id,
            "type": element.type.value,
            "domain": element.domain.value,
            "site": element.site,
        })
        if element.parent_id and element.parent_id in expanded:
            edges.append({"parent": element.parent_id, "child": element.id})
    return {
        "nodes": nodes,
        "edges": edges,
        "note": "Observable topology neighborhood around alarmed elements.",
    }


def _causal_graph(sim, scenario: Scenario) -> dict[str, Any]:
    edges = []
    for alarm in sim.get_alarms():
        edges.append({
            "cause": scenario.element_id,
            "effect": alarm.element_id,
            "evidence": f"alarm:{alarm.name}",
        })
    for sample in sim.get_kpis():
        if sample.is_anomalous:
            edges.append({
                "cause": scenario.element_id,
                "effect": sample.element_id,
                "evidence": f"kpi:{sample.metric}",
            })
    return {
        "root": scenario.element_id,
        "edges": edges[:200],
    }


def _difficulty(scenario: Scenario, alarm_count: int, affected_count: int, topology_size: int) -> str:
    score = 0
    if topology_size > 80:
        score += 2
    if alarm_count >= 12 or affected_count >= 12:
        score += 2
    if scenario.secondary_faults:
        score += 2
    if "distractor_alarms" in scenario.stress_tags:
        score += 1
    if "missing_noisy_telemetry" in scenario.stress_tags:
        score += 1
    if scenario.fault_type in {"FIBER_DEGRADATION", "POWER_BROWNOUT", "GPS_SYNC_LOSS", "UPF_DEGRADATION"}:
        score += 2
    if score >= 5:
        return "hard"
    if score >= 3:
        return "middle"
    return "easy"


if __name__ == "__main__":
    raise SystemExit(main())

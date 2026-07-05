"""Pre-registration generator for frozen synthetic telecom evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .dataset import validate_dataset


DEFAULT_SYSTEMS = [
    "full",
    "single",
    "no_rag",
    "no_consensus",
    "no_arbiter",
    "no_partition",
    "no_debate",
    "no_verifier",
    "no_repair",
]
SUPPORTED_TELCO_SYSTEMS = set(DEFAULT_SYSTEMS)


def build_preregistration(
    dataset_path: str | Path,
    *,
    systems: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    runs: int = 1,
    algorithm_id: str | None = None,
) -> dict[str, Any]:
    path = Path(dataset_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_dataset(data)
    if not validation["ok"]:
        raise ValueError("synthetic dataset validation failed: " + "; ".join(validation["errors"]))
    cases = data.get("cases", [])
    suite = data.get("meta", {}).get("suite", "telco_v3")
    if suite == "telco_v4" and not algorithm_id:
        raise ValueError(
            "telco_v4 requires --algorithm-id (prefer a clean git commit SHA) before preregistration"
        )
    if suite == "telco_v4" and not data.get("meta", {}).get("design", {}).get("seed_source"):
        raise ValueError("telco_v4 dataset is missing auditable seed_source provenance")
    runtime_ids = [case["runtime"]["runtime_case_id"] for case in cases]
    difficulty_counts = Counter(case["labels"]["difficulty"] for case in cases)
    selected_systems = systems or DEFAULT_SYSTEMS
    unsupported = sorted(set(selected_systems) - SUPPORTED_TELCO_SYSTEMS)
    if unsupported:
        raise ValueError(
            "unsupported synthetic telco system(s): "
            + ", ".join(unsupported)
            + ". Supported systems: "
            + ", ".join(DEFAULT_SYSTEMS)
        )
    model_name = model or os.getenv("OPENAI_MODEL") or "configured runtime model"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    max_tool_iters = int(os.getenv("TELCO_MAX_TOOL_ITERS", "6"))
    if suite == "telco_v4" and model_name == "configured runtime model":
        raise ValueError("telco_v4 requires an explicit --model")
    return {
        "status": "frozen",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark": {
            "name": data.get("meta", {}).get("name", "synthetic_telco_v3"),
            "type": "local synthetic telecom RCA fallback",
            "role_in_claim": data.get("meta", {}).get(
                "role_in_claim",
                "last-resort synthetic fallback only; not headline real/gated evidence",
            ),
        },
        "dataset": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "content_sha256": data.get("meta", {}).get("content_sha256"),
            "case_count": len(cases),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "design": data.get("meta", {}).get("design"),
            "validation": validation,
        },
        "row_selection": {
            "method": "all_exported_cases",
            "runtime_case_ids": runtime_ids,
            "source_scenario_ids": [case["labels"]["source_scenario_id"] for case in cases],
            "case_count": len(runtime_ids),
        },
        "systems": selected_systems,
        "algorithm": {
            "id": algorithm_id or "development artifact; not a locked confirmatory algorithm",
            "freeze_rule": (
                "No source, prompt, tool-policy, knowledge-base, or scoring change after this ID "
                "and before all preregistered systems finish."
            ),
        },
        "architecture": {
            "treatment_full": (
                "multi-agent flow with triage, correlation, RAN expert, transport expert, "
                "power/site expert, core expert, consensus, remediation, and validation"
            ),
            "single_baseline": "one unrestricted autonomous NOC engineer with the same primitive tools",
            "diagnostic_principle": "domain decomposition is valid only if wins survive the strong single baseline and ablations",
        },
        "model": {
            "name": model_name,
            "temperature": temperature,
            "base_url": base_url,
            "max_tool_iters": max_tool_iters,
            "cache": False,
        },
        "execution": {
            "runs": runs,
            "independent_unit": "scenario",
            "repeat_handling": (
                "average correctness within scenario for effect estimation; majority vote within "
                "scenario for exact McNemar so repeats are not treated as independent samples"
            ),
            "failure_policy": "retain and report every attempted case; no post-hoc rerun selection",
        },
        "runtime_safety": {
            "runtime_fields": ["runtime_case_id", "incident", "telemetry", "topology", "instructions"],
            "evaluator_only_fields": [
                "root_element_id",
                "acceptable_elements",
                "fault_type",
                "remediation_sop",
                "remediation_keywords",
                "causal_graph",
                "difficulty",
            ],
            "forbidden_runtime_inputs": ["source_scenario_id", "root_element_id", "fault_type", "remediation_sop"],
            "tool_contract": {
                "global_query_kpis": (
                    "returns aggregate anomaly counts by domain/site/metric only; it must not expose "
                    "element IDs or raw KPI readings without an explicit element_id drill-down"
                ),
                "element_query_kpis": "returns detailed KPI readings for the requested element subject to domain access policy",
                "scoring_labels": "used only by evaluator; never passed into agent prompts or tool outputs",
            },
        },
        "metrics": {
            "primary": "strict_diagnosis_accuracy",
            "secondary": [
                "localization_accuracy",
                "fault_type_accuracy",
                "causal_explanation_accuracy",
                "remediation_target_accuracy",
                "remediation_action_accuracy",
                "end_to_end_accuracy",
                "tokens",
                "tool_calls",
                "llm_calls",
                "latency_s",
            ],
        },
        "clear_win_gate": {
            "primary_effect": "full MAS beats the single baseline by >=0.10 absolute strict accuracy",
            "significance": "paired exact p <= 0.05 when sample size permits",
            "joint_rule": "both primary_effect and significance must pass",
            "diagnostics_required": [
                "no_partition",
                "no_debate",
                "no_verifier",
                "no_repair",
                "no_rag",
                "compute accounting",
                "per-difficulty breakdown",
            ],
        },
        "stopping_rule": {
            "fixed_cases": True,
            "fixed_runs": True,
            "no_extension_by_p_value": True,
            "no_post_hoc_filtering": True,
            "no_algorithm_change_after_outcome_inspection": True,
        },
        "commands": [
            f"OPENAI_MODEL={shlex.quote(model_name)} "
            f"OPENAI_BASE_URL={shlex.quote(base_url)} "
            f"TELCO_TEMPERATURE={temperature} "
            f"TELCO_MAX_TOOL_ITERS={max_tool_iters} "
            "python3 -m telco_mas.evaluation.run_benchmark "
            f"--suite {suite} "
            f"--systems {','.join(selected_systems)} "
            f"--runs {runs} --no-cache "
            + (
                f"--synthetic-dataset {shlex.quote(str(path))} --preregistration PREREG_PATH "
                f"--algorithm-id {shlex.quote(str(algorithm_id))} "
                if suite == "telco_v4"
                else ""
            )
            + f"--out results/synthetic_{suite}_benchmark_frozen.json"
        ],
        "note": (
            "Synthetic evidence remains below OpenRCA/TN-RCA/TeleLogsAgent. Telco-v4 may be "
            "called confirmatory only if generated and preregistered before any v4 model outcome."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a synthetic telco-v3 fallback preregistration JSON.")
    parser.add_argument("--dataset", default="results/synthetic_telco_v3_dataset.json")
    parser.add_argument("--out", default="results/prereg_synthetic_telco_v3_frozen.json")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--algorithm-id", default=None,
                        help="required clean commit/artifact ID for confirmatory telco_v4")
    args = parser.parse_args(argv)

    try:
        payload = build_preregistration(
            args.dataset,
            systems=[item.strip() for item in args.systems.split(",") if item.strip()],
            model=args.model,
            temperature=args.temperature,
            runs=args.runs,
            algorithm_id=args.algorithm_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    out = Path(args.out)
    payload["commands"] = [
        command.replace("PREREG_PATH", shlex.quote(str(out)))
        for command in payload["commands"]
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "case_count": payload["dataset"]["case_count"],
        "difficulty_counts": payload["dataset"]["difficulty_counts"],
        "dataset_sha256": payload["dataset"]["sha256"],
    }, indent=2))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

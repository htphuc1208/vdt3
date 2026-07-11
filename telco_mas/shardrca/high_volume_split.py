"""C: a high-volume long-context split where MAS decomposition should help most.

Motivation (Chain-of-Agents, Lost-in-the-Middle): evidence-isolated MAS is
predicted to beat a single context only when the evidence is too large/dispersed
for one context to reliably retain and locate. The mechanism therefore has the
best chance in the *high telemetry volume, multi-modal* regime. This module
selects that regime from the existing label-safe RCAEval hard split, holding out
nothing that a locked prior holdout already consumed, and emits a pre-registration
draft whose stopping rule is justified by an a-priori power analysis (not chosen
after seeing results).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from ..evaluation.stats import mcnemar_exact_power, mcnemar_required_pairs
from .hard_split import build_hard_split


def build_high_volume_split(
    hard_split: dict[str, Any] | None = None,
    *,
    root: str | None = None,
    volume_quantile: float = 0.75,
    require_logs: bool = True,
    require_traces: bool = True,
    exclude_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Select the top telemetry-volume, multi-modal slice of the hard split."""

    payload = hard_split or build_hard_split(root)
    cases = list(payload["cases"])
    exclude = exclude_case_ids or set()
    volumes = sorted(case["telemetry_bytes"] for case in cases)
    if not volumes:
        raise ValueError("empty hard split; cannot build a high-volume split")
    threshold = volumes[min(len(volumes) - 1, int(len(volumes) * volume_quantile))]

    selected = []
    for case in cases:
        if case["runtime_case_id"] in exclude:
            continue
        if case["telemetry_bytes"] < threshold:
            continue
        if require_logs and not case["criteria"]["has_logs"]:
            continue
        if require_traces and not case["criteria"]["has_traces"]:
            continue
        selected.append(case)
    selected.sort(key=lambda case: (-case["telemetry_bytes"], case["runtime_case_id"]))

    # Objective volume bins for a pre-registered per-bin breakdown.
    for case in selected:
        case["volume_bin"] = _volume_bin(case["telemetry_bytes"], volumes)

    return {
        "meta": {
            "suite": "rcaeval_high_volume",
            "derived_from": "rcaeval_hard",
            "selection": (
                f"telemetry_bytes >= p{int(volume_quantile * 100)} of hard split "
                f"AND has_logs={require_logs} AND has_traces={require_traces}"
            ),
            "volume_quantile": volume_quantile,
            "volume_threshold_bytes": threshold,
            "excluded_locked_holdout": sorted(exclude),
            "n_selected": len(selected),
            "label_safety": (
                "runtime_case_id is opaque; selection uses only label-free telemetry-volume "
                "and modality-presence metadata, never root/fault labels"
            ),
        },
        "cases": selected,
    }


def _volume_bin(size_bytes: int, all_volumes: list[int]) -> str:
    p50 = all_volumes[len(all_volumes) // 2]
    p90 = all_volumes[int(len(all_volumes) * 0.9)]
    if size_bytes >= p90:
        return "very_high"
    if size_bytes >= p50:
        return "high"
    return "moderate"


def power_analysis(
    n_pairs: int,
    *,
    alpha: float = 0.05,
    power_target: float = 0.80,
    discordant_rate: float = 0.35,
) -> dict[str, Any]:
    """A-priori power for the frozen n: required n per effect, and MDE at this n."""

    required = {
        f"delta_{int(d * 100)}pp": mcnemar_required_pairs(d, discordant_rate, alpha=alpha, power=power_target)
        for d in (0.10, 0.15, 0.20, 0.25)
    }
    # Minimum detectable effect at the frozen n: smallest marginal delta reaching
    # target power, scanned on the exact test with p10 pinned at a plausible level.
    mde = None
    for delta_pct in range(2, 60, 1):
        delta = delta_pct / 100.0
        p10 = max(0.0, (discordant_rate - delta) / 2.0)
        p01 = p10 + delta
        if p01 + p10 > 1.0:
            break
        result = mcnemar_exact_power(n_pairs, p01=p01, p10=p10, alpha=alpha)
        if (result.get("power") or 0.0) >= power_target:
            mde = {"delta": round(delta, 3), **result}
            break
    return {
        "n_pairs": n_pairs,
        "alpha": alpha,
        "power_target": power_target,
        "assumed_discordant_rate": discordant_rate,
        "required_pairs_by_effect": required,
        "minimum_detectable_effect_at_n": mde,
        "interpretation": (
            "If MDE at this n exceeds the effect the mechanism plausibly produces, "
            "the split cannot confirm a win and a larger benchmark (e.g. TN-RCA530) is required."
        ),
    }


def build_prereg_draft(
    split: dict[str, Any],
    *,
    systems: list[str] | None = None,
    model: str = "${OPENAI_MODEL}",
    temperature: float = 0.0,
    algorithm_id: str = "TBD-freeze-after-weight-fit",
    weights_artifact: str = "results/weights/shardrca_fusion_frozen.json",
) -> dict[str, Any]:
    """Emit a DRAFT pre-registration; it must not be frozen until B/E are locked."""

    systems = systems or [
        "shardrca_full",
        "single_react",
        "single_react_sc",
        "code_retrieval_single",
        "no_interaction",
    ]
    n = split["meta"]["n_selected"]
    case_ids = [case["runtime_case_id"] for case in split["cases"]]
    return {
        "status": "draft",
        "created_at": str(date.today()),
        "purpose": (
            "Confirmatory test of evidence-isolated MAS vs strong single baselines in the "
            "high telemetry-volume, multi-modal regime predicted by Chain-of-Agents."
        ),
        "dataset": {
            "suite": split["meta"]["suite"],
            "derived_from": split["meta"]["derived_from"],
            "selection": split["meta"]["selection"],
            "n": n,
            "excluded_locked_holdout": split["meta"]["excluded_locked_holdout"],
            "runtime_case_ids": case_ids,
        },
        "systems": systems,
        "primary_baseline": "strongest_single (resolved deterministically by the analyzer)",
        "treatment": "shardrca_full",
        "model": model,
        "temperature": temperature,
        "cache": "disabled (--no-cache)",
        "algorithm_id": algorithm_id,
        "fusion_weights_artifact": weights_artifact,
        "primary_metric": "root Hit@1",
        "secondary_metrics": ["Hit@3", "MRR", "tokens", "tool_calls", "latency", "per volume_bin Hit@1"],
        "statistical_test": "exact two-sided paired McNemar, alpha=0.05, Holm across baselines",
        "effect_gate": "absolute Hit@1 delta >= 0.10 OR >= 20% relative error reduction",
        "stopping_rule": "run the frozen case list exactly once; no p-value extension, no post-hoc filtering",
        "power_analysis": power_analysis(n),
        "freeze_preconditions": [
            "default no-fit fusion weights frozen by the preregistration, unless a separate preregistration freezes a disjoint validation artifact before the holdout",
            "algorithm_id set to the clean commit SHA after validation/finalization",
            "MDE at n is at or below the effect the high-volume mechanism is expected to produce",
        ],
        "label_safety": split["meta"]["label_safety"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the high-volume long-context split + prereg draft")
    parser.add_argument("--hard-split", default="results/rcaeval_hard_split.json")
    parser.add_argument("--exclude-prereg", default="results/prereg_v7_holdout20_2026-07-03.json",
                        help="prereg whose runtime_case_ids are a locked holdout to exclude")
    parser.add_argument("--volume-quantile", type=float, default=0.75)
    parser.add_argument("--out-split", default="results/rcaeval_high_volume_split.json")
    parser.add_argument("--out-prereg", default="results/prereg_high_volume_draft.json")
    args = parser.parse_args(argv)

    hard = json.loads(Path(args.hard_split).read_text()) if Path(args.hard_split).exists() else None
    exclude: set[str] = set()
    if Path(args.exclude_prereg).exists():
        locked = json.loads(Path(args.exclude_prereg).read_text())
        exclude = set(locked.get("dataset", {}).get("runtime_case_ids", []))

    split = build_high_volume_split(hard, volume_quantile=args.volume_quantile, exclude_case_ids=exclude)
    prereg = build_prereg_draft(split)
    Path(args.out_split).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_split).write_text(json.dumps(split, indent=2))
    Path(args.out_prereg).write_text(json.dumps(prereg, indent=2))
    print(json.dumps({
        "out_split": args.out_split,
        "out_prereg": args.out_prereg,
        "n_selected": split["meta"]["n_selected"],
        "excluded_locked": len(exclude),
        "power": prereg["power_analysis"]["minimum_detectable_effect_at_n"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

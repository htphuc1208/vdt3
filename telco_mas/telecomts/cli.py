"""Run paired TelecomTS event-level RCA evaluations."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..evaluation.stats import aggregate_ci
from ..llm import LLMClient, extract_json
from ..schemas import UsageStats
from .catalog import CATALOG_POLICY_ID, build_training_catalog, catalog_sha256
from .dataset import RCA_CLASSES, SPLIT_POLICY_ID, TelecomTSDataset, TelecomTSDatasetError
from .evaluator import canonical_class, score_prediction
from .features import KPI_SHARDS, summarize_event
from .prereg import DEFAULT_SYSTEMS, dataset_manifest


METHOD_VERSION = "telecomts_event_shardrca_v4"
MAS_SYSTEM = "telecomts_shardrca_full"
SUPPORTED_SYSTEMS = set(DEFAULT_SYSTEMS)
RADIO_PRIMARY_CLASSES = {
    "Antenna Failure",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)",
    "Doppler Shift (Severe)",
}
RESOURCE_PRIMARY_CLASSES = {
    "Buffer Overflow (Gradual Buildup)",
    "Resource Allocation Bugs",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
}
SPECIALIST_INSTRUCTIONS = {
    "radio_quality": (
        "Analyze only radio signal strength, SNR, BLER, and MCS shape evidence. "
        "Distinguish static level shifts, gradual degradation, and oscillation."
    ),
    "resource_capacity": (
        "Analyze only PRB allocation/utilization and uplink buffer shape evidence. "
        "Distinguish congestion, buffer buildup, and allocation-control faults. "
        "Abrupt or mid-event PRB/utilization spikes with aligned demand favor sudden congestion; "
        "resource allocation bugs require allocation-demand mismatch or control inconsistency."
    ),
    "traffic_protocol": (
        "Analyze only byte volume, packet counts, and protocol transitions. "
        "Distinguish throughput collapse, bursty load, and transport-side symptoms. "
        "Abrupt packet/byte bursts favor sudden congestion; gradual queue-like degradation favors "
        "buffer buildup; traffic collapse alone can be downstream of radio faults."
    ),
}

_SINGLE_SYSTEM_PROMPT = """You are one careful 5G RCA engineer.
Classify exactly one anomaly event from deterministic, label-free KPI summaries.
All candidate classes are listed in the supplied class catalog. Do not invent a class.
If prototype_matches are provided, treat them as weak development-only statistical calibration,
not as a label.
Use the catalog mechanism, primary_evidence, and distinguishers to separate initiating root cause
from downstream symptoms.
Compare temporal shape, cross-layer consistency, and plausible alternatives.
Return ONLY JSON:
{"root_cause":"exact candidate class", "ranked_candidates":["class", "class", "class"],
 "confidence":0.0, "evidence":["short KPI observation"], "rationale":"short explanation"}.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paired TelecomTS event-level RCA evaluation.")
    parser.add_argument("--data-dir", default=os.getenv("TELECOMTS_DATA_DIR") or "data/telecomts")
    parser.add_argument("--prereg", default=None)
    parser.add_argument("--split", choices=["development", "validation", "test"], default="development")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--event-indices", default=None)
    parser.add_argument("--limit", type=int, default=3, help="0 selects all events")
    parser.add_argument("--mode", choices=["profile", "llm"], default="profile")
    parser.add_argument("--algorithm-id", default=None)
    parser.add_argument("--out", default="results/telecomts_development_profile.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        dataset = TelecomTSDataset(args.data_dir)
        prereg = _load_prereg(args.prereg) if args.prereg else None
        if prereg:
            split, event_indices, systems = _verify_prereg(
                dataset,
                prereg,
                mode=args.mode,
                algorithm_id=args.algorithm_id,
                no_cache=args.no_cache,
            )
        else:
            split = args.split
            if split == "test":
                raise ValueError("TelecomTS test split requires a matching frozen preregistration")
            event_indices = (
                _parse_indices(args.event_indices)
                if args.event_indices
                else _first_events(dataset, split, limit=args.limit or None)
            )
            systems = [item.strip() for item in args.systems.split(",") if item.strip()]
            _validate_event_indices(dataset, split, event_indices)
        _validate_systems(systems)
    except (TelecomTSDatasetError, ValueError, OSError, json.JSONDecodeError) as exc:
        _write_skipped(args.out, args, str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    llm = None
    settings = get_settings()
    if args.mode == "llm":
        if not settings.has_api_key and not args.cache_only:
            print("ERROR: --mode llm requires OPENAI_API_KEY unless --cache-only is used.", file=sys.stderr)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    manifest_sha256 = dataset_manifest(dataset)["sha256"]
    training_catalog = build_training_catalog(dataset)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "suite": "telecomts",
        "benchmark": "telecomts",
        "status": "running",
        "mode": args.mode,
        "method_version": METHOD_VERSION,
        "event_unit": True,
        "split_policy_id": SPLIT_POLICY_ID,
        "split": split,
        "event_indices": event_indices,
        "event_ids": [dataset.events(split)[index].event_id for index in event_indices],
        "systems": systems,
        "prereg": args.prereg,
        "algorithm_id": args.algorithm_id,
        "data_dir": str(dataset.root_dir),
        "dataset_manifest_sha256": manifest_sha256,
        "training_catalog_policy_id": CATALOG_POLICY_ID,
        "training_catalog_sha256": catalog_sha256(training_catalog),
        "model": settings.model if args.mode == "llm" else None,
        "temperature": settings.temperature if args.mode == "llm" else None,
        "cache_enabled": bool(llm.cache_enabled) if llm is not None else False,
        "evidence_warning": _evidence_warning(args.mode, split),
    }
    try:
        rows = _resume_rows(args.out, meta) if args.resume else []
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if split == "test" and Path(args.out).exists() and not args.resume:
        print("ERROR: refusing to overwrite an existing TelecomTS test result; use --resume", file=sys.stderr)
        return 2

    completed = {(str(row.get("system")), str(row.get("case_id"))) for row in rows}
    for event_index in event_indices:
        task = dataset.get_runtime_event(split, event_index)
        board = summarize_event(task["payload"], class_catalog=training_catalog, include_prototype_matches=True)
        for system in systems:
            key = (system, task["case_id"])
            if key in completed:
                continue
            row = _run_one(
                dataset,
                task,
                board,
                split=split,
                event_index=event_index,
                system=system,
                mode=args.mode,
                llm=llm,
            )
            rows.append(row)
            completed.add(key)
            _write_payload(args.out, meta, rows)
            print(
                f"[{system}] {split} event#{event_index} "
                f"strict={row['strict_correct']} prediction={row.get('predicted')}"
            )

    meta["status"] = "completed"
    payload = _write_payload(args.out, meta, rows)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _run_one(
    dataset: TelecomTSDataset,
    task: dict[str, Any],
    board: dict[str, Any],
    *,
    split: str,
    event_index: int,
    system: str,
    mode: str,
    llm: LLMClient | None,
) -> dict[str, Any]:
    started = time.time()
    error = None
    try:
        if mode == "profile":
            prediction = {
                "root_cause": None,
                "ranked_candidates": [],
                "rationale": "profile-mode ingestion smoke; no diagnosis generated",
            }
            usage = UsageStats()
            diagnostics: dict[str, Any] = {"board_shards": list(board["shards"])}
        else:
            assert llm is not None
            prediction, usage, diagnostics = _llm_prediction(system, board, llm)
    except Exception as exc:  # keep paired checkpoints complete; failures score as incorrect
        error = f"{type(exc).__name__}: {exc}"
        prediction = {"root_cause": None, "ranked_candidates": [], "rationale": error}
        usage = UsageStats()
        diagnostics = {}
    score = score_prediction(dataset.event_target(split, event_index), prediction)
    return {
        "case_id": task["case_id"],
        "split": split,
        "event_index": event_index,
        "system": system,
        "source_window_count": len(dataset.events(split)[event_index].records),
        "sample_length": task["payload"]["sample_length"],
        "prediction": prediction,
        "diagnostics": diagnostics,
        "error": error,
        "latency_s": round(time.time() - started, 3),
        **_usage_dict(usage),
        **score.to_dict(),
    }


def _llm_prediction(
    system: str,
    board: dict[str, Any],
    llm: LLMClient,
) -> tuple[dict[str, Any], UsageStats, dict[str, Any]]:
    if system == MAS_SYSTEM:
        return _mas_prediction(board, llm)
    samples = 1
    if system == "single_react_sc":
        samples = 3
    elif system == "single_equal_calls":
        samples = 5
    prompt = dict(board)
    if system == "same_board_single":
        prompt["instruction"] = "Cross-check all three deterministic shard summaries before deciding."
    candidates, usage = _sample_single(prompt, llm, samples=samples, system=system)
    winner = _vote(candidates)
    return winner, usage, {"independent_samples": candidates, "sample_count": samples}


def _sample_single(
    board: dict[str, Any],
    llm: LLMClient,
    *,
    samples: int,
    system: str,
) -> tuple[list[dict[str, Any]], UsageStats]:
    usage = UsageStats()
    candidates = []
    for index in range(samples):
        payload = {"independent_replicate": index + 1, "evidence_board": board}
        response = llm.chat(
            [
                {"role": "system", "content": _SINGLE_SYSTEM_PROMPT},
                {"role": "user", "content": _compact_json(payload)},
            ],
            force_json=True,
        )
        usage = usage.add(response.usage)
        candidate = extract_json(response.content)
        candidate["proposed_by"] = f"{system}:{index + 1}"
        candidates.append(candidate)
    return candidates, usage


def _mas_prediction(
    board: dict[str, Any],
    llm: LLMClient,
) -> tuple[dict[str, Any], UsageStats, dict[str, Any]]:
    usage = UsageStats()
    generalist_samples, generalist_usage = _sample_single(
        board,
        llm,
        samples=1,
        system="telecomts_mas_generalist",
    )
    usage = usage.add(generalist_usage)
    generalist = generalist_samples[0] if generalist_samples else {}
    findings = []
    for shard_name in KPI_SHARDS:
        specialist_payload = {
            "scenario": board["scenario"],
            "sample_length": board["sample_length"],
            "sampling_rate_hz": board["sampling_rate_hz"],
            "class_catalog": board["class_catalog"],
            "prototype_matches": board["prototype_matches"]["by_shard"].get(shard_name, []),
            "visible_shard": shard_name,
            "kpi_summaries": board["shards"][shard_name],
        }
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"You are the {shard_name} specialist in a 5G RCA team. "
                        f"{SPECIALIST_INSTRUCTIONS[shard_name]} You cannot see other shards. "
                        "If prototype matches are provided, use only your shard-specific matches "
                        "as weak calibration. "
                        "Use catalog primary_evidence and distinguishers for classes visible from your shard. "
                        "Do not prematurely collapse uncertainty from your partial view. Return ONLY JSON with "
                        "observations (KPI-grounded list), class_scores (all ten exact class names mapped to 0..1), "
                        "ranked_candidates (top 5), confidence, uncertainty, and rationale."
                    ),
                },
                {"role": "user", "content": _compact_json(specialist_payload)},
            ],
            force_json=True,
        )
        usage = usage.add(response.usage)
        findings.append({
            "specialist": shard_name,
            "finding": extract_json(response.content),
        })

    adjudicator_response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are the final adjudicator in a 5G RCA team. You receive one full-board "
                    "generalist assessment and three shard-isolated specialist assessments. Treat "
                    "visibility as reliability: a specialist can falsify or qualify evidence in its "
                    "own shard, but partial-shard consensus must not override the full-board diagnosis "
                    "unless it identifies direct contradictions. Separate root cause from downstream "
                    "symptoms; radio or hardware faults can cause later resource and traffic degradation. "
                    "When a shard specialist has direct primary evidence for its domain, require the "
                    "generalist to have equally direct contradictory evidence before overriding it. "
                    "Do not override a radio or hardware diagnosis when the radio_quality specialist "
                    "also ranks that radio/hardware class high; resource symptoms may be downstream. "
                    "If the generalist chooses a radio/hardware class but the radio_quality specialist "
                    "does not support it, and the resource_capacity or traffic_protocol specialist has "
                    "direct high-confidence primary evidence, prefer the domain specialist. "
                    "For resource faults, distinguish allocation-demand mismatch from aligned offered-load spikes. "
                    "If prototype matches are provided, use them as weak development-only statistical "
                    "calibration, not labels. "
                    "Use class mechanisms, primary_evidence, distinguishers, and every class-score vector. "
                    "Return ONLY JSON "
                    "with root_cause, ranked_candidates (top 5), confidence, evidence, "
                    "disagreements, generalist_assessment, specialist_assessment, and rationale. "
                    "Use exact candidate class names."
                ),
            },
            {"role": "user", "content": _compact_json({
                "class_catalog": board["class_catalog"],
                "generalist_assessment": generalist,
                "specialist_findings": findings,
                "evidence_board": board,
            })},
        ],
        force_json=True,
    )
    usage = usage.add(adjudicator_response.usage)
    adjudication = extract_json(adjudicator_response.content)
    finalists = _finalists(adjudication, [{"finding": generalist}, *findings], limit=5)
    verified = canonical_class(adjudication.get("root_cause"))
    if verified is None:
        generalist_root = canonical_class(generalist.get("root_cause"))
        if generalist_root:
            adjudication["root_cause"] = generalist_root
            adjudication["adjudicator_contract_fallback"] = "generalist_root_cause"
        else:
            adjudication["root_cause"] = finalists[0]
            adjudication["adjudicator_contract_fallback"] = "ranked_candidate"
    else:
        adjudication["root_cause"] = verified
    ranked = adjudication.get("ranked_candidates")
    if not isinstance(ranked, list):
        adjudication["ranked_candidates"] = finalists
    guard = _apply_domain_guard(adjudication, generalist, findings, board)
    return adjudication, usage, {
        "generalist_assessment": generalist,
        "specialist_findings": findings,
        "adjudication": adjudication,
        "finalists": finalists,
        "domain_guard": guard,
        "call_count": 5,
    }


def _apply_domain_guard(
    prediction: dict[str, Any],
    generalist: dict[str, Any],
    findings: list[dict[str, Any]],
    board: dict[str, Any],
) -> dict[str, Any]:
    before = canonical_class(prediction.get("root_cause"))
    guard = {"applied": False, "before": before, "after": before, "reason": None}
    radio = _finding_for(findings, "radio_quality")
    resource = _finding_for(findings, "resource_capacity")
    generalist_root = canonical_class(generalist.get("root_cause"))

    # Guard 1: If generalist chose a radio/hardware class and the radio
    # specialist confirms it, trust that diagnosis even if the adjudicator
    # was swayed by downstream resource symptoms.
    if generalist_root in RADIO_PRIMARY_CLASSES:
        radio_score = _class_score(radio, generalist_root)
        radio_rank = _rank_index(radio, generalist_root)
        generalist_confidence = _confidence(generalist)
        if radio_rank in {0, 1} and (radio_score >= 0.2 or generalist_confidence >= 0.75):
            _set_guarded_prediction(
                prediction,
                generalist_root,
                guard,
                reason="generalist_radio_class_confirmed_by_radio_specialist",
            )
            return guard
        # Guard 2: If radio specialist doesn't support any radio class but
        # resource specialist has very strong evidence, prefer the resource diagnosis.
        resource_top, resource_score = _top_scored_class(resource, RESOURCE_PRIMARY_CLASSES)
        if (
            radio_score < 0.3
            and _max_class_score(radio, RADIO_PRIMARY_CLASSES) < 0.3
            and _best_rank_in(radio, RADIO_PRIMARY_CLASSES) not in {0, 1}
            and resource_top
            and resource_score >= 0.8
        ):
            _set_guarded_prediction(
                prediction,
                resource_top,
                guard,
                reason="weak_radio_support_and_strong_resource_specialist_evidence",
            )
            return guard

    # Guard 3: If adjudicator chose a resource class but the radio specialist
    # has strong evidence for a radio class, override to the radio class.
    # This corrects cases where downstream resource symptoms mislead the adjudicator.
    if before in RESOURCE_PRIMARY_CLASSES or before is None:
        radio_top = _rank_index(radio, None)  # unused, get top class instead
        radio_ranked = radio.get("ranked_candidates", [])
        if isinstance(radio_ranked, list) and radio_ranked:
            top_radio = canonical_class(radio_ranked[0])
            if top_radio in RADIO_PRIMARY_CLASSES:
                top_score = _class_score(radio, top_radio)
                if top_score >= 0.6 and generalist_root == top_radio:
                    _set_guarded_prediction(
                        prediction,
                        top_radio,
                        guard,
                        reason="radio_specialist_and_generalist_agree_on_radio_class",
                    )
                    return guard

    # Guard 4: Aligned abrupt resource+traffic load with no relief signature is a
    # sudden congestion spike, not a resource-allocation bug or radio fault.
    if _has_sudden_load_signature(board):
        _set_guarded_prediction(
            prediction,
            "High Network Congestion (Sudden Spike)",
            guard,
            reason="abrupt_resource_and_traffic_load_signature",
        )
        return guard

    # Guard 5: Cross-layer abrupt + oscillatory churn (radio, resource, traffic all
    # oscillating with drops) is the handover-thrash signature.
    if _has_handover_churn_signature(board):
        _set_guarded_prediction(
            prediction,
            "Faulty Handover Algorithm (Too Frequent)",
            guard,
            reason="cross_layer_abrupt_oscillatory_churn_signature",
        )
        return guard

    # Guard 6: Aligned gradual resource + traffic buildup with no relief is gradual
    # congestion, distinct from a radio/hardware fault.
    if _has_gradual_load_buildup_signature(board):
        _set_guarded_prediction(
            prediction,
            "High Network Congestion (Gradual Buildup)",
            guard,
            reason="aligned_resource_and_traffic_buildup_signature",
        )
        return guard

    return guard


def _radio_strongly_confirms(radio: dict[str, Any], class_name: str | None) -> bool:
    if class_name not in RADIO_PRIMARY_CLASSES:
        return False
    return _rank_index(radio, class_name) == 0 and _class_score(radio, class_name) >= 0.3


def _set_guarded_prediction(
    prediction: dict[str, Any],
    root_cause: str,
    guard: dict[str, Any],
    *,
    reason: str,
) -> None:
    prediction["root_cause"] = root_cause
    ranked = prediction.get("ranked_candidates")
    if not isinstance(ranked, list):
        ranked = []
    ranked = [root_cause, *[item for item in ranked if canonical_class(item) != root_cause]]
    prediction["ranked_candidates"] = ranked[:5]
    guard.update({
        "applied": True,
        "after": root_cause,
        "reason": reason,
    })


def _finding_for(findings: list[dict[str, Any]], specialist: str) -> dict[str, Any]:
    for item in findings:
        if item.get("specialist") == specialist and isinstance(item.get("finding"), dict):
            return item["finding"]
    return {}


def _class_score(finding: dict[str, Any], class_name: str) -> float:
    scores = finding.get("class_scores")
    if not isinstance(scores, dict):
        return 0.0
    try:
        return float(scores.get(class_name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _confidence(prediction: dict[str, Any]) -> float:
    try:
        return float(prediction.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rank_index(finding: dict[str, Any], class_name: str) -> int | None:
    ranked = finding.get("ranked_candidates")
    if not isinstance(ranked, list):
        return None
    for index, value in enumerate(ranked):
        if canonical_class(value) == class_name:
            return index
    return None


def _top_scored_class(
    finding: dict[str, Any],
    allowed_classes: set[str],
) -> tuple[str | None, float]:
    candidates = [
        (class_name, _class_score(finding, class_name))
        for class_name in allowed_classes
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0] if candidates else (None, 0.0)


def _max_class_score(finding: dict[str, Any], allowed_classes: set[str]) -> float:
    return max((_class_score(finding, class_name) for class_name in allowed_classes), default=0.0)


def _best_rank_in(finding: dict[str, Any], allowed_classes: set[str]) -> int | None:
    ranks = [
        rank
        for class_name in allowed_classes
        if (rank := _rank_index(finding, class_name)) is not None
    ]
    return min(ranks) if ranks else None


def _has_sudden_load_signature(board: dict[str, Any]) -> bool:
    resource = _shape_tag_counts(board, "resource_capacity")
    traffic = _shape_tag_counts(board, "traffic_protocol")
    return (
        resource["abrupt_step"] >= 5
        and traffic["abrupt_step"] >= 3
        and resource["falling_or_downward"] == 0
        and traffic["falling_or_downward"] == 0
    )


def _has_handover_churn_signature(board: dict[str, Any]) -> bool:
    radio = _shape_tag_counts(board, "radio_quality")
    resource = _shape_tag_counts(board, "resource_capacity")
    traffic = _shape_tag_counts(board, "traffic_protocol")
    return (
        radio["abrupt_step"] >= 1
        and resource["abrupt_step"] >= 5
        and traffic["abrupt_step"] >= 3
        and resource["oscillatory"] >= 4
        and traffic["oscillatory"] >= 2
        and resource["falling_or_downward"] >= 2
        and traffic["falling_or_downward"] >= 1
    )


def _has_gradual_load_buildup_signature(board: dict[str, Any]) -> bool:
    resource = _shape_tag_counts(board, "resource_capacity")
    traffic = _shape_tag_counts(board, "traffic_protocol")
    return (
        resource["upward_or_rising"] >= 2
        and resource["mid_event_spike"] >= 3
        and traffic["mid_event_spike"] >= 2
        and resource["falling_or_downward"] == 0
        and traffic["falling_or_downward"] == 0
    )


def _shape_tag_counts(board: dict[str, Any], shard_name: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    shard = (board.get("shards") or {}).get(shard_name) or {}
    if not isinstance(shard, dict):
        return counts
    for summary in shard.values():
        if not isinstance(summary, dict) or summary.get("kind") != "numeric":
            continue
        tags = summary.get("shape_tags")
        if not isinstance(tags, list):
            continue
        normalized = {str(tag) for tag in tags}
        counts.update(normalized)
        if "falling_edge" in normalized or "downward_trend" in normalized:
            counts["falling_or_downward"] += 1
        if "rising_edge" in normalized or "upward_trend" in normalized:
            counts["upward_or_rising"] += 1
    return counts


def _finalists(
    fusion: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    limit: int = 2,
) -> list[str]:
    candidates = []
    ranked_values = fusion.get("ranked_candidates")
    if not isinstance(ranked_values, list):
        ranked_values = []
    for value in [fusion.get("root_cause"), *ranked_values]:
        canonical = canonical_class(value)
        if canonical and canonical not in candidates:
            candidates.append(canonical)
    if len(candidates) < 2:
        voted = _vote([item["finding"] for item in findings])
        for value in voted.get("ranked_candidates", []):
            canonical = canonical_class(value)
            if canonical and canonical not in candidates:
                candidates.append(canonical)
    for name in RCA_CLASSES:
        if len(candidates) >= limit:
            break
        if name not in candidates:
            candidates.append(name)
    return candidates[:limit]


def _vote(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    votes: Counter[str] = Counter()
    confidence: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        root = canonical_class(candidate.get("root_cause"))
        if root is None:
            ranked = candidate.get("ranked_candidates")
            root = canonical_class(ranked[0]) if isinstance(ranked, list) and ranked else None
        if root is None:
            continue
        votes[root] += 1
        try:
            confidence[root] += float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            pass
        if isinstance(candidate.get("evidence"), list):
            evidence[root].extend(str(item) for item in candidate["evidence"][:3])
    if not votes:
        return {
            "root_cause": None,
            "ranked_candidates": [],
            "confidence": 0.0,
            "rationale": "No valid candidate class was returned.",
            "vote_breakdown": {},
        }
    ranked = sorted(votes, key=lambda name: (-votes[name], -confidence[name], name))
    winner = ranked[0]
    return {
        "root_cause": winner,
        "ranked_candidates": ranked,
        "confidence": round(confidence[winner] / max(1, votes[winner]), 4),
        "evidence": evidence[winner][:8],
        "rationale": f"Independent-sample vote selected {winner}.",
        "vote_breakdown": dict(votes),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary = aggregate_ci(rows, ["strict_correct", "total_tokens", "llm_calls", "latency_s"])
    for system in sorted({str(row["system"]) for row in rows}):
        system_rows = [row for row in rows if row["system"] == system]
        per_class = {}
        for name in RCA_CLASSES:
            class_rows = [row for row in system_rows if row.get("target") == name]
            per_class[name] = (
                round(sum(1 for row in class_rows if row.get("strict_correct")) / len(class_rows), 4)
                if class_rows
                else None
            )
        observed = [value for value in per_class.values() if value is not None]
        summary[system]["macro_accuracy"] = round(sum(observed) / len(observed), 4) if observed else 0.0
        summary[system]["per_class_accuracy"] = per_class
        summary[system]["errors"] = sum(1 for row in system_rows if row.get("error"))
    return summary


def _verify_prereg(
    dataset: TelecomTSDataset,
    prereg: dict[str, Any],
    *,
    mode: str,
    algorithm_id: str | None,
    no_cache: bool,
) -> tuple[str, list[int], list[str]]:
    selection = prereg.get("event_selection", {})
    split = selection.get("split")
    indices = selection.get("event_indices")
    event_ids = selection.get("event_ids")
    if split not in dataset.splits or not isinstance(indices, list) or not indices:
        raise ValueError("TelecomTS preregistration has no valid event selection")
    _validate_event_indices(dataset, split, indices)
    current_ids = [dataset.events(split)[index].event_id for index in indices]
    if event_ids != current_ids:
        raise ValueError("TelecomTS preregistered event IDs differ from current data")
    if prereg.get("dataset", {}).get("manifest_sha256") != dataset_manifest(dataset)["sha256"]:
        raise ValueError("TelecomTS dataset manifest differs from preregistration")
    training_catalog = build_training_catalog(dataset)
    expected_catalog = prereg.get("training_catalog", {})
    if expected_catalog.get("policy_id") != CATALOG_POLICY_ID:
        raise ValueError("TelecomTS training catalog policy differs from preregistration")
    if expected_catalog.get("sha256") != catalog_sha256(training_catalog):
        raise ValueError("TelecomTS development-only training catalog differs from preregistration")
    systems = prereg.get("systems")
    if not isinstance(systems, list):
        raise ValueError("TelecomTS preregistration has no systems")
    _validate_systems(systems)
    if split == "test":
        if prereg.get("status") != "frozen":
            raise ValueError("TelecomTS test preregistration is not frozen")
        expected_algorithm = str(prereg.get("algorithm", {}).get("id") or "")
        if not algorithm_id or algorithm_id != expected_algorithm:
            raise ValueError("--algorithm-id must match the frozen TelecomTS preregistration")
        if mode != "llm":
            raise ValueError("TelecomTS test split can only be consumed by the preregistered LLM run")
        settings = get_settings()
        execution = prereg.get("execution", {})
        if execution.get("model") != settings.model:
            raise ValueError("OPENAI_MODEL differs from frozen TelecomTS preregistration")
        if float(execution.get("temperature")) != settings.temperature:
            raise ValueError("TELCO_TEMPERATURE differs from frozen TelecomTS preregistration")
        if execution.get("cache") is not False or not no_cache:
            raise ValueError("Frozen TelecomTS test run requires --no-cache")
    return str(split), list(indices), list(systems)


def _validate_systems(systems: list[str]) -> None:
    if not systems or len(set(systems)) != len(systems):
        raise ValueError("TelecomTS systems must be non-empty and unique")
    unknown = [system for system in systems if system not in SUPPORTED_SYSTEMS]
    if unknown:
        raise ValueError(f"Unsupported TelecomTS systems: {', '.join(unknown)}")


def _validate_event_indices(dataset: TelecomTSDataset, split: str, indices: list[int]) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError("TelecomTS event indices must be unique")
    count = len(dataset.events(split))
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"TelecomTS event index outside {split} range 0..{count - 1}")


def _resume_rows(path: str | Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.exists():
        return []
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    existing = payload.get("meta", {})
    for key in (
        "suite",
        "mode",
        "method_version",
        "split_policy_id",
        "split",
        "event_indices",
        "event_ids",
        "systems",
        "algorithm_id",
        "dataset_manifest_sha256",
        "training_catalog_policy_id",
        "training_catalog_sha256",
        "model",
        "temperature",
        "cache_enabled",
    ):
        if existing.get(key) != meta.get(key):
            raise ValueError(f"Cannot resume TelecomTS result: metadata mismatch for {key}")
    return list(payload.get("rows") or [])


def _write_payload(path: str | Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {"meta": meta, "summary": _summary(rows), "rows": rows}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_skipped(path: str | Path, args: argparse.Namespace, reason: str) -> None:
    if Path(path).exists():
        return
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "telecomts",
            "benchmark": "telecomts",
            "status": "skipped",
            "reason": reason,
            "mode": args.mode,
            "split": args.split,
            "prereg": args.prereg,
        },
        "summary": {},
        "rows": [],
    }
    _write_json(path, payload)


def _first_events(dataset: TelecomTSDataset, split: str, *, limit: int | None) -> list[int]:
    indices = list(range(len(dataset.events(split))))
    return indices if limit is None else indices[: min(limit, len(indices))]


def _parse_indices(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_prereg(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _usage_dict(usage: UsageStats) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "llm_calls": usage.llm_calls,
        "tool_calls": usage.tool_calls,
    }


def _evidence_warning(mode: str, split: str) -> str:
    if mode == "profile":
        return "profile mode is ingestion-only and cannot support a result claim"
    if split != "test":
        return "development/validation output is calibration evidence and cannot support a result claim"
    return "TelecomTS is testbed-backed synthetic RCA and can only support an explicit synthetic-only claim"


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

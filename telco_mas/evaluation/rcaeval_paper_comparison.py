"""Compare ShardRCA with the official WWW'25 RCAEval BARO reproduction."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from .stats import paired_bootstrap_effect, paired_mcnemar, wilson_ci


def compare(
    shard_result: str | Path,
    checkpoint_dir: str | Path,
    baro_result_root: str | Path,
) -> dict[str, Any]:
    shard_payload = json.loads(Path(shard_result).read_text(encoding="utf-8"))
    checkpoint_root = Path(checkpoint_dir) / "shardrca_full"
    baro_root = Path(baro_result_root)
    rows = []
    missing = []

    for shard_row in shard_payload.get("rows", []):
        case_id = str(shard_row["case_id"])
        checkpoint = checkpoint_root / f"{_safe_case_id(case_id)}.json"
        baro_result = baro_root / _baro_filename(case_id)
        if not checkpoint.is_file() or not baro_result.is_file():
            missing.append({
                "case_id": case_id,
                "checkpoint": str(checkpoint),
                "baro_result": str(baro_result),
            })
            continue

        prediction = json.loads(checkpoint.read_text(encoding="utf-8"))
        shard_ranks = _dedupe(
            shard_row.get("ranked_roots")
            or [prediction.get("root"), *(prediction.get("ranked_roots") or [])]
        )
        baro_indicators = json.loads(baro_result.read_text(encoding="utf-8")).get("0", [])
        baro_ranks = _dedupe(
            str(indicator).split("_", 1)[0].replace("-db", "")
            for indicator in baro_indicators
        )
        truth = str(shard_row["true_root"])
        fault = _fault_type(case_id)
        rows.extend([
            _row(
                case_id,
                "rcaeval_www25_baro",
                baro_ranks,
                truth,
                fault,
            ),
            _row(
                case_id,
                "rcaeval_shardrca_full",
                shard_ranks,
                truth,
                fault,
                usage=prediction,
            ),
        ])

    if missing:
        raise ValueError(f"Missing paired artifacts: {json.dumps(missing, indent=2)}")

    systems = ("rcaeval_www25_baro", "rcaeval_shardrca_full")
    summary = {system: _summary(rows, system) for system in systems}
    by_fault = {}
    for fault in sorted({str(row["fault_type"]) for row in rows}):
        fault_rows = [row for row in rows if row["fault_type"] == fault]
        by_fault[fault] = {system: _summary(fault_rows, system) for system in systems}

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "benchmark": "RCAEval WWW'25 RE2 paired paper comparison",
            "paper_baseline": "BARO",
            "paper_repository": "https://github.com/phamquiluan/RCAEval",
            "paper_branch": "www25",
            "paper_commit": "9d14687ce0644188f1f1a576fd3f57cd903af446",
            "shard_result": str(shard_result),
            "baro_result_root": str(baro_result_root),
            "paired_cases": len(rows) // 2,
        },
        "summary": summary,
        "by_fault_type": by_fault,
        "paired_tests": {
            "hit_at_1": paired_mcnemar(
                rows,
                "rcaeval_www25_baro",
                "rcaeval_shardrca_full",
                "hit_at_1",
            ),
            "avg_at_5": paired_bootstrap_effect(
                rows,
                "rcaeval_www25_baro",
                "rcaeval_shardrca_full",
                "avg_at_5",
                samples=10_000,
                seed=20260706,
            ),
        },
        "rows": rows,
    }


def _row(
    case_id: str,
    system: str,
    ranks: list[str],
    truth: str,
    fault_type: str,
    *,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    rank = next((index for index, root in enumerate(ranks, 1) if root == truth), None)
    return {
        "case_id": case_id,
        "system": system,
        "true_root": truth,
        "predicted_root": ranks[0] if ranks else "UNKNOWN",
        "ranked_roots": ranks,
        "fault_type": fault_type,
        "hit_at_1": rank == 1,
        "hit_at_3": rank is not None and rank <= 3,
        "hit_at_5": rank is not None and rank <= 5,
        "avg_at_5": (6 - rank) / 5 if rank is not None and rank <= 5 else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "llm_calls": int(usage.get("llm_calls") or 0),
        "tool_calls": int(usage.get("tool_calls") or 0),
        "latency_s": float(usage.get("latency_s") or 0.0),
    }


def _summary(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    selected = [row for row in rows if row["system"] == system]
    n = len(selected)
    hit1 = sum(bool(row["hit_at_1"]) for row in selected)
    _, lo, hi = wilson_ci(hit1, n)
    return {
        "n": n,
        "ac_at_1": round(hit1 / n, 4) if n else 0.0,
        "ac_at_1_ci95": [lo, hi],
        "ac_at_3": round(mean(row["hit_at_3"] for row in selected), 4) if n else 0.0,
        "ac_at_5": round(mean(row["hit_at_5"] for row in selected), 4) if n else 0.0,
        "avg_at_5": round(mean(row["avg_at_5"] for row in selected), 4) if n else 0.0,
        "mrr": round(mean(row["mrr"] for row in selected), 4) if n else 0.0,
        "total_tokens": sum(row["total_tokens"] for row in selected),
        "llm_calls": sum(row["llm_calls"] for row in selected),
        "tool_calls": sum(row["tool_calls"] for row in selected),
        "latency_s": round(sum(row["latency_s"] for row in selected), 2),
    }


def _baro_filename(case_id: str) -> str:
    prefix = "RCAEval-RE2-TT-"
    if not case_id.startswith(prefix):
        raise ValueError(f"Expected an RE2-TT case ID, got: {case_id}")
    fault_label, repetition = case_id[len(prefix):].rsplit("-", 1)
    return f"{fault_label}_{repetition}.json"


def _fault_type(case_id: str) -> str:
    fault_label = case_id.rsplit("-", 1)[0]
    return fault_label.rsplit("_", 1)[-1]


def _dedupe(values) -> list[str]:
    output = []
    for value in values:
        text = str(value or "")
        if text and text not in output:
            output.append(text)
    return output


def _safe_case_id(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-result", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--baro-result-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = compare(args.shard_result, args.checkpoint_dir, args.baro_result_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "paired_cases": payload["meta"]["paired_cases"],
        "summary": payload["summary"],
        "paired_tests": payload["paired_tests"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

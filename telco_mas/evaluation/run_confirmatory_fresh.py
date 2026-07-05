"""Run the frozen fresh-holdout RCAEval-hard confirmatory (prereg_rcaeval_fresh_confirm24_frozen).

Treatment: evidence-isolated MAS (shardrca_full) with the frozen causal re-rank weights.
Primary baseline: budgeted single-context ReAct with self-consistency (single_react_sc).
Boundary control (reported, not gated): same_board_single global-board oracle.

Per-case checkpoints make the run resumable and idempotent. The case list, systems,
weights, and stopping rule are pinned by the frozen preregistration; this runner only
executes and scores it.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from ..llm import LLMClient
from ..config import get_settings
from ..evaluation.stats import accuracy_at_k, paired_mcnemar, reciprocal_rank, wilson_ci
from ..shardrca.hard_split import load_hard_cases_by_runtime_ids
from ..shardrca.runner import run_rcaeval_case

PREREG = "results/prereg_rcaeval_fresh_confirm24_frozen.json"
SYSTEMS = ["shardrca_full", "single_react", "single_react_sc", "same_board_single"]


def _checkpoint_path(ckpt_dir: Path, system: str, case_id: str) -> Path:
    safe = case_id.replace("/", "_")
    return ckpt_dir / system / f"{safe}.json"


def _score_row(pred, case, system: str) -> dict:
    ranked = pred.ranked_roots or [pred.root]
    return {
        "case_id": case.case_id,
        "runtime_case_id": case.runtime_case_id(),
        "system": f"rcaeval_{system}",
        "predicted_root": pred.root,
        "true_root": case.ground_truth_root,
        "hit_at_1": int(accuracy_at_k(ranked, case.ground_truth_root, 1)),
        "hit_at_3": int(accuracy_at_k(ranked, case.ground_truth_root, 3)),
        "mrr": round(reciprocal_rank(ranked, case.ground_truth_root), 4),
        "root_hit_at_1": int(accuracy_at_k(ranked, case.ground_truth_root, 1)),
        "total_tokens": pred.total_tokens,
        "llm_calls": pred.llm_calls,
        "tool_calls": pred.tool_calls,
        "latency_s": pred.latency_s,
        "notes": pred.notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", default=PREREG)
    parser.add_argument("--out", default="results/rcaeval_hard_llm_fresh_confirm24.json")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/rcaeval_fresh_confirm24")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    args = parser.parse_args(argv)

    prereg = json.load(open(args.prereg))
    ids = prereg["dataset"]["runtime_case_ids"]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    settings = get_settings()
    if not settings.has_api_key:
        print("ERROR: live LLM run requires OPENAI_API_KEY", flush=True)
        return 2
    llm = LLMClient(cache_enabled=False)

    cases, missing = load_hard_cases_by_runtime_ids(ids)
    if missing:
        print(f"ERROR: missing runtime ids: {missing}", flush=True)
        return 2

    ckpt_dir = Path(args.checkpoint_dir)
    rows = []
    for system in systems:
        for case in cases:
            cpath = _checkpoint_path(ckpt_dir, system, case.case_id)
            if cpath.exists():
                row = json.load(open(cpath))
            else:
                pred = run_rcaeval_case(case, system=system, llm=llm)
                row = _score_row(pred, case, system)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(f"[{system:18s}] {case.case_id[:44]:44s} hit1={row['hit_at_1']} "
                  f"pred={str(row['predicted_root'])[:22]:22s} true={row['true_root']}", flush=True)

    # summary
    summary = {}
    for system in systems:
        skey = f"rcaeval_{system}"
        srows = [r for r in rows if r["system"] == skey]
        n = len(srows)
        h1 = sum(r["hit_at_1"] for r in srows)
        summary[skey] = {
            "n": n,
            "hit_at_1": round(h1 / n, 4) if n else 0.0,
            "hit_at_1_ci95": [round(x, 4) for x in wilson_ci(h1, n)[:2]] if n else [0, 0],
            "hit_at_3": round(sum(r["hit_at_3"] for r in srows) / n, 4) if n else 0.0,
            "mrr": round(sum(r["mrr"] for r in srows) / n, 4) if n else 0.0,
            "total_tokens": sum(r["total_tokens"] for r in srows),
        }

    baseline = "rcaeval_single_react_sc"
    treatment = "rcaeval_shardrca_full"
    mcnemar = paired_mcnemar(rows, baseline, treatment, "hit_at_1")
    b_h1 = summary[baseline]["hit_at_1"]
    t_h1 = summary[treatment]["hit_at_1"]
    abs_delta = round(t_h1 - b_h1, 4)
    rel_err_red = round((b_h1 - t_h1) / (1 - b_h1), 4) if b_h1 < 1 else 0.0
    # Note: relative error reduction here is for the treatment reducing baseline error
    rel_err_red = round(((1 - b_h1) - (1 - t_h1)) / (1 - b_h1), 4) if b_h1 < 1 else 0.0
    gate_pass = (
        (abs_delta >= 0.10 or rel_err_red >= 0.20)
        and mcnemar["p_value_exact"] <= 0.05
        and mcnemar["treatment_only_correct"] > mcnemar["baseline_only_correct"]
    )
    clear_win_gate = {
        "passed": bool(gate_pass),
        "required": "treatment beats budgeted single_react_sc by >=0.10 abs Hit@1 or >=20% rel error reduction, exact paired p<=0.05",
        "baseline": baseline,
        "treatment": treatment,
        "baseline_hit_at_1": b_h1,
        "treatment_hit_at_1": t_h1,
        "absolute_delta": abs_delta,
        "relative_error_reduction": rel_err_red,
        "mcnemar": mcnemar,
    }

    # boundary vs oracle (reported, not gated)
    oracle_boundary = None
    if "rcaeval_same_board_single" in summary:
        oracle_boundary = {
            "note": "global-board oracle boundary; NOT part of the confirmatory gate",
            "same_board_single_hit_at_1": summary["rcaeval_same_board_single"]["hit_at_1"],
            "shardrca_full_hit_at_1": t_h1,
            "mcnemar": paired_mcnemar(rows, "rcaeval_same_board_single", treatment, "hit_at_1"),
        }

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prereg": args.prereg,
            "suite": "rcaeval_hard_llm_fresh_confirm",
            "model": settings.model,
            "systems": systems,
            "shardrca_weights": os.environ.get("SHARDRCA_WEIGHTS", "(default)"),
        },
        "summary": summary,
        "clear_win_gate": clear_win_gate,
        "oracle_boundary": oracle_boundary,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    for skey, s in summary.items():
        print(f"  {skey:30s} Hit@1={s['hit_at_1']:.3f} Hit@3={s['hit_at_3']:.3f} MRR={s['mrr']:.3f} tok={s['total_tokens']}")
    print(f"\nCLEAR-WIN GATE (vs {baseline}): passed={clear_win_gate['passed']} "
          f"delta={abs_delta:+.3f} p={mcnemar['p_value_exact']} "
          f"(t_wins={mcnemar['treatment_only_correct']}, b_wins={mcnemar['baseline_only_correct']})")
    if oracle_boundary:
        ob = oracle_boundary
        print(f"ORACLE BOUNDARY: same_board={ob['same_board_single_hit_at_1']:.3f} vs MAS={ob['shardrca_full_hit_at_1']:.3f} "
              f"p={ob['mcnemar']['p_value_exact']}")
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

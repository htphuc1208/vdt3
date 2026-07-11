"""Group-A confirmatory harness: close the budget/mechanism confounds.

This runner answers the reviewer's decisive questions that the headline
RCAEval-Hard comparison left open:

  1. Compute-parity: does ShardRCA still beat a *same-budget* single-context
     agent? Baseline ``single_equal_tokens`` gives one ReAct agent an expanded
     tool budget so its token/tool spend is comparable to the MAS.
  2. Mechanism isolation: is the gain from *evidence isolation + peer
     interaction*, or merely from ensembling / more compute?
       - ``no_shard``       = one serial reader over all modalities (isolates the
                              value of sharding vs a single global read).
       - ``no_interaction`` = independent shard workers fused by log-opinion pool
                              with NO peer critique (isolates the interaction round
                              from plain ensembling).
  3. Metric-prior separation: report Hit@1 with the label-safe metric prior ON
     and OFF, so the "augmented" variant is never conflated with the core system.

Every system is evaluated *paired* on one frozen, label-safe holdout (opaque
runtime ids), with per-case checkpoints so the live run is resumable. For every
baseline we report an exact paired McNemar test and a scenario-level paired
bootstrap effect against ``shardrca_full``.

Use ``--dry-run`` (llm=None) to validate the harness deterministically at zero
API cost before spending on a live run.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean

from ..config import get_settings
from ..llm import LLMClient
from ..shardrca.hard_split import (
    build_hard_split,
    load_hard_cases,
    load_hard_cases_by_runtime_ids,
)
from ..shardrca.runner import run_rcaeval_case
from .external import ExternalBenchmarkCase, ExternalPrediction
from .rcaeval_adapter import load_cases, validate_rcaeval
from .stats import (
    accuracy_at_k,
    paired_bootstrap_effect,
    paired_mcnemar,
    reciprocal_rank,
    wilson_ci,
)

DEFAULT_TREATMENT = "shardrca_full"
DEFAULT_SYSTEMS = [
    "shardrca_full",       # treatment
    "single_react_sc",     # original (budget-limited) baseline
    "single_equal_tokens",  # compute-parity control
    "no_shard",            # ablation: single serial reader (isolates sharding)
    "no_interaction",      # ablation: independent workers + PoE, no peer critique
]


# --------------------------------------------------------------------------- #
# Frozen, label-safe holdout                                                   #
# --------------------------------------------------------------------------- #
def build_or_load_prereg(
    prereg_path: Path,
    *,
    n: int,
    seed: int,
    source_filter: set[str],
    rcaeval_root: str | None,
) -> tuple[list[ExternalBenchmarkCase], dict]:
    """Load a frozen holdout if it exists, else deterministically build+freeze one.

    The frozen artifact stores only opaque ``runtime_case_id`` hashes, never
    ground-truth labels or label-derived paths, so pinning the confirmatory case
    list never leaks answers.
    """

    if prereg_path.exists():
        prereg = json.loads(prereg_path.read_text())
        ids = prereg["dataset"]["runtime_case_ids"]
        cases, missing = load_hard_cases_by_runtime_ids(ids, rcaeval_root)
        if missing:
            raise SystemExit(f"frozen prereg references missing runtime ids: {missing}")
        return cases, prereg

    all_cases = load_hard_cases(rcaeval_root, sample=None, seed=seed)
    if source_filter:
        all_cases = [c for c in all_cases if c.source in source_filter]
    if not all_cases:
        raise SystemExit(f"no hard cases for source filter {sorted(source_filter)}")
    import random

    if n and n < len(all_cases):
        chosen = random.Random(seed).sample(all_cases, n)
    else:
        chosen = list(all_cases)
    chosen = sorted(chosen, key=lambda c: c.case_id)
    split_meta = build_hard_split(rcaeval_root)["meta"]
    prereg = {
        "meta": {
            "frozen_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "rcaeval_hard_group_a",
            "purpose": "compute-parity + mechanism-isolation confirmatory holdout",
            "seed": seed,
            "requested_n": n,
            "actual_n": len(chosen),
            "source_filter": sorted(source_filter),
            "selection": split_meta.get("selection"),
            "label_safety": split_meta.get("label_safety"),
        },
        "dataset": {
            "runtime_case_ids": [c.runtime_case_id() for c in chosen],
        },
    }
    prereg_path.parent.mkdir(parents=True, exist_ok=True)
    prereg_path.write_text(json.dumps(prereg, indent=2))
    print(f"[prereg] froze {len(chosen)} hard cases -> {prereg_path}", flush=True)
    return chosen, prereg


def build_or_load_reuse_prereg(
    prereg_path: Path,
    *,
    reuse_result: Path,
    source_filter: set[str],
    rcaeval_root: str | None,
) -> tuple[list[ExternalBenchmarkCase], dict]:
    """Pin the exact engineering case ids used by a prior result (e.g. the BARO
    RE2-TT comparison), so the metric-prior ablation is directly interpretable
    against that published table rather than a fresh sample.

    Unlike the hard-split prereg, this intentionally records engineering case ids
    (which encode the fault family) because the whole point is to *reproduce a
    named subset*; it is a reproduction pin, not a blind confirmatory holdout.
    """

    if prereg_path.exists():
        prereg = json.loads(prereg_path.read_text())
        wanted = list(prereg["dataset"]["engineering_case_ids"])
    else:
        prior = json.loads(reuse_result.read_text())
        wanted = sorted({str(r["case_id"]) for r in prior.get("rows", []) if r.get("case_id")})
        prereg = None

    all_cases = load_cases(rcaeval_root, sources=source_filter or None)
    by_id = {c.case_id: c for c in all_cases}
    missing = [cid for cid in wanted if cid not in by_id]
    if missing:
        raise SystemExit(f"reuse subset references missing case ids: {missing[:5]} ...")
    cases = sorted((by_id[cid] for cid in wanted), key=lambda c: c.case_id)

    if prereg is None:
        prereg = {
            "meta": {
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "suite": "rcaeval_re2tt_group_a3_metric_prior",
                "purpose": "metric-prior separation on the exact BARO RE2-TT subset",
                "reused_from": str(reuse_result),
                "actual_n": len(cases),
                "label_safety": "reproduction pin: engineering ids recorded to mirror a named comparison",
            },
            "dataset": {"engineering_case_ids": wanted},
        }
        prereg_path.parent.mkdir(parents=True, exist_ok=True)
        prereg_path.write_text(json.dumps(prereg, indent=2))
        print(f"[prereg] pinned {len(cases)} reuse cases -> {prereg_path}", flush=True)
    return cases, prereg


# --------------------------------------------------------------------------- #
# Per-case run + scoring                                                        #
# --------------------------------------------------------------------------- #
def _score_row(pred: ExternalPrediction, case: ExternalBenchmarkCase, system: str, prior: str) -> dict:
    ranked = pred.ranked_roots or [pred.root]
    return {
        "case_id": case.case_id,
        "runtime_case_id": case.runtime_case_id(),
        "source": case.source,
        "system": system,
        "prior": prior,
        "predicted_root": pred.root,
        "true_root": case.ground_truth_root,
        "hit_at_1": int(accuracy_at_k(ranked, case.ground_truth_root, 1)),
        "hit_at_3": int(accuracy_at_k(ranked, case.ground_truth_root, 3)),
        "mrr": round(reciprocal_rank(ranked, case.ground_truth_root), 4),
        "total_tokens": pred.total_tokens,
        "llm_calls": pred.llm_calls,
        "tool_calls": pred.tool_calls,
        "latency_s": pred.latency_s,
    }


def _checkpoint_path(ckpt_dir: Path, prior: str, system: str, case_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in case_id)
    return ckpt_dir / f"prior_{prior}" / system / f"{safe}.json"


def run_matrix(
    cases: list[ExternalBenchmarkCase],
    systems: list[str],
    prior_settings: list[str],
    *,
    llm: LLMClient | None,
    ckpt_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    for prior in prior_settings:
        apply_prior = prior == "on"
        for system in systems:
            for idx, case in enumerate(cases, 1):
                cpath = _checkpoint_path(ckpt_dir, prior, system, case.case_id)
                if cpath.exists():
                    row = json.loads(cpath.read_text())
                else:
                    pred = run_rcaeval_case(case, system=system, llm=llm, apply_prior=apply_prior)
                    row = _score_row(pred, case, system, prior)
                    cpath.parent.mkdir(parents=True, exist_ok=True)
                    cpath.write_text(json.dumps(row, indent=2))
                rows.append(row)
                print(
                    f"[prior={prior:3s}][{system:20s}] {idx:>3}/{len(cases)} "
                    f"{case.runtime_case_id():18s} hit1={row['hit_at_1']} "
                    f"pred={str(row['predicted_root'])[:20]:20s} true={row['true_root']}",
                    flush=True,
                )
    return rows


# --------------------------------------------------------------------------- #
# Analysis                                                                       #
# --------------------------------------------------------------------------- #
def _system_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    h1 = sum(r["hit_at_1"] for r in rows)
    center, lo, hi = wilson_ci(h1, n)
    return {
        "n": n,
        "hit_at_1": round(h1 / n, 4),
        "hit_at_1_ci95": [lo, hi],
        "hit_at_3": round(sum(r["hit_at_3"] for r in rows) / n, 4),
        "mrr": round(mean(r["mrr"] for r in rows), 4),
        "mean_total_tokens": round(mean(r["total_tokens"] for r in rows), 1),
        "mean_tool_calls": round(mean(r["tool_calls"] for r in rows), 2),
        "mean_llm_calls": round(mean(r["llm_calls"] for r in rows), 2),
        "mean_latency_s": round(mean(r["latency_s"] for r in rows), 2),
    }


def _pairwise(rows: list[dict], baseline: str, treatment: str = DEFAULT_TREATMENT) -> dict:
    """Exact paired McNemar + paired bootstrap effect for baseline vs treatment."""
    mcnemar = paired_mcnemar(rows, baseline, treatment, "hit_at_1")
    effect = paired_bootstrap_effect(rows, baseline, treatment, "hit_at_1")
    return {
        "baseline": baseline,
        "treatment": treatment,
        "delta_hit_at_1_mean": effect["mean_difference"],
        "delta_hit_at_1_ci95": effect["mean_difference_ci95"],
        "paired_cases": mcnemar["paired_cases"],
        "treatment_only_correct": mcnemar["treatment_only_correct"],
        "baseline_only_correct": mcnemar["baseline_only_correct"],
        "p_value_exact": mcnemar["p_value_exact"],
    }


def analyze(
    rows: list[dict],
    systems: list[str],
    prior_settings: list[str],
    *,
    treatment: str = DEFAULT_TREATMENT,
) -> dict:
    out: dict = {"by_prior": {}}
    for prior in prior_settings:
        prior_rows = [r for r in rows if r["prior"] == prior]
        summaries = {s: _system_summary([r for r in prior_rows if r["system"] == s]) for s in systems}
        pairwise = {
            s: _pairwise(prior_rows, s, treatment)
            for s in systems
            if s != treatment
        }
        # Compute-parity ratio: treatment tokens / baseline tokens.
        t_tok = summaries.get(treatment, {}).get("mean_total_tokens", 0) or 0
        parity = {}
        for s in systems:
            b_tok = summaries.get(s, {}).get("mean_total_tokens", 0) or 0
            parity[s] = round(t_tok / b_tok, 3) if b_tok else None
        out["by_prior"][prior] = {
            "summary": summaries,
            "pairwise_vs_treatment": pairwise,
            "treatment_token_ratio_over_baseline": parity,
        }

    # Metric-prior effect: shardrca_full & the single baseline, ON vs OFF.
    if set(prior_settings) >= {"on", "off"}:
        effect = {}
        for s in {treatment, "single_react_sc", "single_equal_tokens"} & set(systems):
            on = [dict(r, system="prior_on") for r in rows if r["system"] == s and r["prior"] == "on"]
            off = [dict(r, system="prior_off") for r in rows if r["system"] == s and r["prior"] == "off"]
            mc = paired_mcnemar(on + off, "prior_off", "prior_on", "hit_at_1")
            effect[s] = {
                "hit_at_1_prior_off": round(mean(r["hit_at_1"] for r in off), 4) if off else None,
                "hit_at_1_prior_on": round(mean(r["hit_at_1"] for r in on), 4) if on else None,
                "prior_only_gain_cases": mc["treatment_only_correct"],
                "prior_only_loss_cases": mc["baseline_only_correct"],
                "p_value_exact": mc["p_value_exact"],
            }
        out["metric_prior_effect"] = effect
    return out


# --------------------------------------------------------------------------- #
# CLI                                                                            #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="holdout size (ignored if prereg exists)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument(
        "--treatment",
        default=DEFAULT_TREATMENT,
        help="system treated as the primary method in paired comparisons",
    )
    parser.add_argument("--source", default=None, help="optional RCAEval source filter, e.g. RE2-TT")
    parser.add_argument("--prior", choices=["on", "off", "both"], default="off",
                        help="metric-prior setting; 'both' runs on+off for the prior-effect table")
    parser.add_argument("--prereg", default="results/prereg_group_a_frozen.json")
    parser.add_argument("--reuse-result", default=None,
                        help="pin the exact engineering case ids from a prior result JSON "
                             "(e.g. the BARO RE2-TT run) instead of a hard-split sample")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/group_a")
    parser.add_argument("--out", default="results/group_a_confirmatory.json")
    parser.add_argument("--rcaeval-root", default=None)
    parser.add_argument("--dry-run", action="store_true", help="llm=None deterministic run (no API cost)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(argv)

    validation = validate_rcaeval(args.rcaeval_root)
    if not validation["ok"]:
        print(f"RCAEval validation failed: {json.dumps(validation, indent=2)}", flush=True)
        return 2

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    treatment = args.treatment.strip()
    if not treatment:
        raise SystemExit("--treatment must name a system")
    if treatment not in systems:
        systems = [treatment, *systems]
    prior_settings = ["off", "on"] if args.prior == "both" else [args.prior]
    source_filter = {s.strip() for s in str(args.source or "").split(",") if s.strip()}

    if args.reuse_result:
        cases, prereg = build_or_load_reuse_prereg(
            Path(args.prereg),
            reuse_result=Path(args.reuse_result),
            source_filter=source_filter,
            rcaeval_root=args.rcaeval_root,
        )
    else:
        cases, prereg = build_or_load_prereg(
            Path(args.prereg),
            n=args.n,
            seed=args.seed,
            source_filter=source_filter,
            rcaeval_root=args.rcaeval_root,
        )
    print(f"[holdout] n={len(cases)} systems={systems} prior={prior_settings}", flush=True)

    llm = None
    if not args.dry_run:
        settings = get_settings()
        if not settings.has_api_key:
            print("ERROR: live run requires OPENAI_API_KEY (or use --dry-run).", flush=True)
            return 2
        llm = LLMClient(cache_enabled=not args.no_cache, cache_only=args.cache_only)

    rows = run_matrix(
        cases, systems, prior_settings, llm=llm, ckpt_dir=Path(args.checkpoint_dir)
    )
    analysis = analyze(rows, systems, prior_settings, treatment=treatment)

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "suite": "rcaeval_hard_group_a",
            "mode": "dry_run_deterministic" if args.dry_run else "live_llm",
            "model": None if args.dry_run else get_settings().model,
            "systems": systems,
            "treatment": treatment,
            "prior_settings": prior_settings,
            "prereg": args.prereg,
            "holdout_n": len(cases),
            "validation": validation,
        },
        "analysis": analysis,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    _print_report(analysis, systems, prior_settings, treatment=treatment)
    print(f"\nSaved -> {out_path}", flush=True)
    return 0


def _print_report(
    analysis: dict,
    systems: list[str],
    prior_settings: list[str],
    *,
    treatment: str = DEFAULT_TREATMENT,
) -> None:
    for prior in prior_settings:
        block = analysis["by_prior"][prior]
        print(f"\n=== prior={prior} ===", flush=True)
        print(f"{'system':22s} {'Hit@1':>7s} {'Hit@3':>7s} {'MRR':>6s} {'tok':>8s} {'tokΔ':>6s}")
        for s in systems:
            summ = block["summary"].get(s, {})
            if not summ.get("n"):
                continue
            ratio = block["treatment_token_ratio_over_baseline"].get(s)
            print(
                f"{s:22s} {summ['hit_at_1']:7.3f} {summ['hit_at_3']:7.3f} "
                f"{summ['mrr']:6.3f} {summ['mean_total_tokens']:8.0f} "
                f"{('%.2f' % ratio) if ratio else '  -  ':>6s}"
            )
        print(f"  paired vs {treatment} (exact McNemar):", flush=True)
        for s, pw in block["pairwise_vs_treatment"].items():
            print(
                f"    {s:22s} Δ={pw['delta_hit_at_1_mean']:+.3f} "
                f"CI{pw['delta_hit_at_1_ci95']} p={pw['p_value_exact']} "
                f"(t_wins={pw['treatment_only_correct']}, b_wins={pw['baseline_only_correct']})",
                flush=True,
            )
    if "metric_prior_effect" in analysis:
        print("\n=== metric-prior effect (ON vs OFF) ===", flush=True)
        for s, e in analysis["metric_prior_effect"].items():
            print(
                f"  {s:22s} off={e['hit_at_1_prior_off']} on={e['hit_at_1_prior_on']} "
                f"(prior_only_gain={e['prior_only_gain_cases']}, loss={e['prior_only_loss_cases']}, "
                f"p={e['p_value_exact']})",
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())

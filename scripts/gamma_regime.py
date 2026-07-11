"""Does evidence-isolated sharding add value in the over-context regime (gamma>1)?

This is the decisive test of ShardRCA's central thesis. Group-A showed that with a
generous budget (gamma<=1) a single holistic reader matches the sharded pipeline
(shardrca_llmboard == no_shard), because all discriminative evidence fits. The
paper's own theory only justifies decomposition when gamma = N(x)/B_eff > 1 and
the decision evidence is DISTRIBUTED across shards.

We instantiate exactly the paper's claim ("fused shard messages retain more mutual
information about Y than a SAME-BUDGET global summary") as a same-budget A/B:

  Both systems share one identical pool of deterministic findings (the union of
  component-group shard miners) and the SAME LLM synthesis decision engine. The
  ONLY difference is how a fixed budget of B findings is allocated before the LLM
  sees them:

    single_budgeted(B): the GLOBAL top-B findings by score (a noisy region can
                        crowd the true root's evidence out of the budget).
    shard_budgeted(B):  round-robin across shards, so every shard (region) is
                        guaranteed representation within the same B.

Sweeping B traces the regime: at small B (gamma>1) shard allocation should retain
the root's evidence that global truncation drops; at large B (gamma<=1) they
converge. Everything downstream of allocation is identical, so any gap is
attributable to evidence isolation, not to more compute or a different decoder.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from telco_mas.config import get_settings
from telco_mas.llm import LLMClient
from telco_mas.evaluation.stats import accuracy_at_k, normalize_id, paired_mcnemar, wilson_ci
from telco_mas.shardrca.board import Blackboard
from telco_mas.shardrca.catalog import build_catalog_for_case, make_component_group_shards
from telco_mas.shardrca.hard_split import build_hard_split, load_hard_cases_by_runtime_ids
from telco_mas.shardrca.miner import MinerWorker
from telco_mas.shardrca.synthesizer import synthesize


def _shard_finding_pools(case, *, finding_limit=60, chunksize=50_000):
    """Return (per-shard ranked finding lists, union pool, distinct component count)."""
    catalog = build_catalog_for_case(case, compute_ranges=False)
    shards = make_component_group_shards(catalog, group_size=6)
    worker = MinerWorker(limit=finding_limit, chunksize=chunksize)
    pools = []
    union = []
    for result in worker.run_many(shards):
        b = Blackboard(case_id=catalog.case_id)
        b.extend(result.findings)
        ranked = b.ranked_findings()
        if ranked:
            pools.append(ranked)
            union.extend(ranked)
    union_board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
    union_board.extend(union)
    components = len({f.component for f in union_board.ranked_findings() if f.component})
    return pools, union_board, components


def _single_budget(union_board, B):
    return union_board.ranked_findings(B)


def _shard_budget(pools, B):
    """Round-robin across shard pools until B findings collected (fair allocation)."""
    cursors = [list(p) for p in pools]
    out = []
    while len(out) < B and any(cursors):
        progressed = False
        for c in cursors:
            if c:
                out.append(c.pop(0))
                progressed = True
                if len(out) >= B:
                    break
        if not progressed:
            break
    return out[:B]


def _decide(case, findings, catalog_summary, llm, *, B):
    board = Blackboard(case_id=case.case_id, catalog_summary=catalog_summary)
    board.extend(findings)
    synth = synthesize(board, llm=llm, k=1, max_findings=B)
    return synth.winner.component, synth.usage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=24, help="widest hard cases to use")
    ap.add_argument("--budgets", default="4,8,16,32")
    ap.add_argument("--prereg", default="results/prereg_gamma_regime.json")
    ap.add_argument("--out", default="results/gamma_regime.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    # Preregister the widest (label-free: by metric column count) hard cases.
    if Path(args.prereg).exists():
        ids = json.loads(Path(args.prereg).read_text())["dataset"]["runtime_case_ids"]
    else:
        rows = build_hard_split(None)["cases"]
        rows = sorted(rows, key=lambda r: (-r["metric_columns"], -r["telemetry_bytes"], r["runtime_case_id"]))
        ids = [r["runtime_case_id"] for r in rows[: args.n]]
        Path(args.prereg).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prereg).write_text(json.dumps(
            {"meta": {"suite": "gamma_regime", "selection": "widest hard cases by metric_columns (label-free)",
                      "n": len(ids)}, "dataset": {"runtime_case_ids": ids}}, indent=2))
        print(f"[prereg] froze {len(ids)} widest hard cases -> {args.prereg}", flush=True)

    cases, missing = load_hard_cases_by_runtime_ids(ids)
    if missing:
        raise SystemExit(f"missing ids: {missing}")

    llm = None
    if not args.dry_run:
        if not get_settings().has_api_key:
            print("ERROR: live run needs OPENAI_API_KEY (or --dry-run)", flush=True)
            return 2
        llm = LLMClient(cache_enabled=True)

    rows = []
    for i, case in enumerate(sorted(cases, key=lambda c: c.case_id), 1):
        pools, union_board, ncomp = _shard_finding_pools(case)
        nfind = len(union_board.ranked_findings())
        truth = case.ground_truth_root
        tn = normalize_id(truth)
        for B in budgets:
            single_f = _single_budget(union_board, B)
            shard_f = _shard_budget(pools, B)
            # decoder-free information retention: is the true root present in-budget?
            single_has = int(any(normalize_id(f.component) == tn for f in single_f))
            shard_has = int(any(normalize_id(f.component) == tn for f in shard_f))
            s_root, _ = _decide(case, single_f, union_board.catalog_summary, llm, B=B)
            h_root, _ = _decide(case, shard_f, union_board.catalog_summary, llm, B=B)
            rows.append({
                "case_id": case.case_id, "source": case.source, "budget": B,
                "n_findings": nfind, "n_components": ncomp,
                "gamma_findings": round(nfind / B, 3), "gamma_components": round(ncomp / B, 3),
                "true_root": truth,
                "single_pred": s_root, "shard_pred": h_root,
                "single_has_root": single_has, "shard_has_root": shard_has,
                "single_hit": int(accuracy_at_k([s_root], truth, 1)),
                "shard_hit": int(accuracy_at_k([h_root], truth, 1)),
            })
        print(f"[{i}/{len(cases)}] {case.runtime_case_id()} findings={nfind} comps={ncomp}", flush=True)

    # Analysis: per-budget Hit@1 + paired McNemar (single vs shard).
    per_budget = {}
    for B in budgets:
        br = [r for r in rows if r["budget"] == B]
        n = len(br)
        s = sum(r["single_hit"] for r in br); h = sum(r["shard_hit"] for r in br)
        mc_rows = []
        for r in br:
            mc_rows.append({"case_id": r["case_id"], "system": "single", "hit_at_1": r["single_hit"]})
            mc_rows.append({"case_id": r["case_id"], "system": "shard", "hit_at_1": r["shard_hit"]})
        mc = paired_mcnemar(mc_rows, "single", "shard", "hit_at_1")
        s_ret = sum(r["single_has_root"] for r in br)
        h_ret = sum(r["shard_has_root"] for r in br)
        per_budget[B] = {
            "n": n,
            "single_root_retention": round(s_ret / n, 4) if n else 0.0,
            "shard_root_retention": round(h_ret / n, 4) if n else 0.0,
            "single_hit_at_1": round(s / n, 4) if n else 0.0,
            "shard_hit_at_1": round(h / n, 4) if n else 0.0,
            "single_ci95": list(wilson_ci(s, n)[1:]) if n else [0, 0],
            "shard_ci95": list(wilson_ci(h, n)[1:]) if n else [0, 0],
            "delta": round((h - s) / n, 4) if n else 0.0,
            "shard_only_correct": mc["treatment_only_correct"],
            "single_only_correct": mc["baseline_only_correct"],
            "p_value_exact": mc["p_value_exact"],
            "mean_gamma_findings": round(sum(r["gamma_findings"] for r in br) / n, 2) if n else 0.0,
        }

    payload = {
        "meta": {"suite": "gamma_regime", "mode": "dry_run" if args.dry_run else "live",
                 "model": None if args.dry_run else get_settings().model,
                 "budgets": budgets, "n_cases": len(cases), "prereg": args.prereg},
        "per_budget": per_budget, "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("\n=== gamma regime: same-budget single vs shard allocation ===", flush=True)
    print("root retention (decoder-free MI proxy) + LLM Hit@1:")
    print(f"{'B':>4} {'gamma':>7} | {'ret_single':>10} {'ret_shard':>9} | {'h_single':>8} {'h_shard':>7} {'delta':>7} {'p':>8}")
    for B in budgets:
        e = per_budget[B]
        print(f"{B:>4} {e['mean_gamma_findings']:>7} | {e['single_root_retention']:>10.3f} {e['shard_root_retention']:>9.3f} | "
              f"{e['single_hit_at_1']:>8.3f} {e['shard_hit_at_1']:>7.3f} {e['delta']:>+7.3f} {e['p_value_exact']:>8}", flush=True)
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

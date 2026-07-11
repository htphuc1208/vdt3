"""Fit the topology/temporal rerank strength on a DISJOINT hard-split validation
set, then evaluate the chosen strength ONCE on the frozen Group-A holdout.

Rationale (Hướng-2): the Group-A diagnosis showed shardrca_full's dominant error
is picking a topology-adjacent symptom carrier (caller/hub) over the true root
(callee/leaf). The topology reranker exists but ships disabled (gamma=0). Here we
fit gamma honestly on cases NOT in the holdout, then report the holdout result a
single time, so this is not tuning-on-the-test.

Because peer interaction is empirically inert (live shardrca_full == offline
no_interaction, 50/50), the whole pipeline is evaluated deterministically at zero
API cost, and the numbers are faithful to the live system.

The per-case board + fused candidates + dependency graph are computed once and
cached; only the final rerank+falsify is re-run per gamma, so the sweep is cheap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from telco_mas.evaluation.stats import accuracy_at_k
from telco_mas.evaluation.stats import paired_mcnemar
from telco_mas.shardrca.catalog import build_catalog_for_case, make_component_group_shards
from telco_mas.shardrca.board import Blackboard
from telco_mas.shardrca.falsifier import falsify
from telco_mas.shardrca.fusion import candidate_evidence_from_findings, fuse_candidate_evidence
from telco_mas.shardrca.hard_split import load_hard_cases, load_hard_cases_by_runtime_ids
from telco_mas.shardrca.miner import MinerWorker
from telco_mas.shardrca.runner import _rcaeval_trace_graph, run_single_baseline
from telco_mas.shardrca.topology import topology_rerank
from telco_mas.shardrca.temporal import temporal_rerank


def _pipeline_state(case, *, finding_limit=25, chunksize=50_000):
    """Deterministic board + fused candidates + graph + symptomatic set for a case."""
    catalog = build_catalog_for_case(case, compute_ranges=False)
    shards = make_component_group_shards(catalog, group_size=6)
    board = Blackboard(case_id=catalog.case_id, catalog_summary=catalog.summary())
    worker = MinerWorker(limit=finding_limit, chunksize=chunksize)
    evidence = []
    for result in worker.run_many(shards):
        board.extend(result.findings)
        evidence.extend(candidate_evidence_from_findings(result.shard_id, result.findings))
    synthesis = fuse_candidate_evidence(evidence, board, max_candidates=8)
    graph = _rcaeval_trace_graph(catalog)
    symptomatic = {c for c, s in board.top_components(20) if s > 0}
    return board, synthesis, graph, symptomatic, case.ground_truth_root


def _winner_component(board, synthesis, graph, symptomatic, *, gamma, beta):
    reranked = topology_rerank(synthesis, symptomatic, graph, gamma=gamma, max_candidates=8)
    reranked = temporal_rerank(reranked, board, beta=beta, max_candidates=8)
    result = falsify(board, reranked.candidates, top=reranked.winner)
    return result.winner.component


def _hit1(component, truth) -> int:
    return int(accuracy_at_k([component], truth, 1))


def evaluate(states, *, gamma, beta) -> float:
    hits = sum(_hit1(_winner_component(*s[:4], gamma=gamma, beta=beta), s[4]) for s in states)
    return hits / len(states) if states else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-prereg", default="results/prereg_group_a_frozen.json")
    ap.add_argument("--fit-n", type=int, default=40)
    ap.add_argument("--fit-seed", type=int, default=13)
    ap.add_argument("--gammas", default="0,0.5,1,2,4,8")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--out", default="results/topo_fit.json")
    args = ap.parse_args(argv)

    holdout_ids = set(json.loads(Path(args.holdout_prereg).read_text())["dataset"]["runtime_case_ids"])
    holdout_cases, missing = load_hard_cases_by_runtime_ids(list(holdout_ids))
    if missing:
        raise SystemExit(f"missing holdout ids: {missing}")

    # Disjoint fit set: all hard cases minus holdout, deterministic sample.
    import random
    all_cases = [c for c in load_hard_cases(None, sample=None, seed=7)
                 if c.runtime_case_id() not in holdout_ids]
    fit_cases = sorted(random.Random(args.fit_seed).sample(all_cases, min(args.fit_n, len(all_cases))),
                       key=lambda c: c.case_id)
    print(f"[fit] {len(fit_cases)} disjoint cases; [holdout] {len(holdout_cases)} cases", flush=True)

    gammas = [float(g) for g in args.gammas.split(",") if g.strip()]

    print("[fit] building deterministic pipeline states...", flush=True)
    fit_states = []
    for i, c in enumerate(fit_cases, 1):
        fit_states.append(_pipeline_state(c))
        if i % 10 == 0:
            print(f"  fit {i}/{len(fit_cases)}", flush=True)

    # Sweep gamma on the fit set.
    fit_curve = {}
    for g in gammas:
        acc = evaluate(fit_states, gamma=g, beta=args.beta)
        fit_curve[g] = round(acc, 4)
        print(f"[fit] gamma={g:<4} beta={args.beta} Hit@1={acc:.3f}", flush=True)
    best_gamma = max(gammas, key=lambda g: (fit_curve[g], -g))  # prefer smaller gamma on ties

    # Evaluate the chosen gamma ONCE on the frozen holdout, plus references.
    print(f"\n[holdout] chosen gamma={best_gamma} (fit Hit@1={fit_curve[best_gamma]})", flush=True)
    print("[holdout] building deterministic pipeline states...", flush=True)
    holdout_states = []
    for i, c in enumerate(sorted(holdout_cases, key=lambda c: c.case_id), 1):
        holdout_states.append((c, _pipeline_state(c)))
        if i % 10 == 0:
            print(f"  holdout {i}/{len(holdout_cases)}", flush=True)

    rows = []
    for c, st in holdout_states:
        board, synthesis, graph, symptomatic, truth = st
        base_component = _winner_component(board, synthesis, graph, symptomatic, gamma=0.0, beta=0.0)
        topo_component = _winner_component(board, synthesis, graph, symptomatic, gamma=best_gamma, beta=args.beta)
        # no_shard reference (single serial reader + self-consistency + falsifier), deterministic
        ns = run_single_baseline(build_catalog_for_case(c, compute_ranges=False), llm=None, self_consistency=True)
        ns_result = falsify(ns.board, ns.synthesis.candidates, top=ns.synthesis.winner)
        rows.append({
            "case_id": c.case_id, "source": c.source, "true_root": truth,
            "no_interaction_base": base_component, "shardrca_topo": topo_component,
            "no_shard": ns_result.winner.component,
            "hit_base": _hit1(base_component, truth),
            "hit_topo": _hit1(topo_component, truth),
            "hit_noshard": _hit1(ns_result.winner.component, truth),
        })

    n = len(rows)
    summary = {
        "no_interaction_base": round(sum(r["hit_base"] for r in rows) / n, 4),
        "shardrca_topo": round(sum(r["hit_topo"] for r in rows) / n, 4),
        "no_shard": round(sum(r["hit_noshard"] for r in rows) / n, 4),
        "n": n,
    }
    # paired McNemar: topo vs base, and topo vs no_shard
    mc_rows = []
    for r in rows:
        mc_rows.append({"case_id": r["case_id"], "system": "base", "hit_at_1": r["hit_base"]})
        mc_rows.append({"case_id": r["case_id"], "system": "topo", "hit_at_1": r["hit_topo"]})
        mc_rows.append({"case_id": r["case_id"], "system": "noshard", "hit_at_1": r["hit_noshard"]})
    mc_topo_vs_base = paired_mcnemar(mc_rows, "base", "topo", "hit_at_1")
    mc_topo_vs_noshard = paired_mcnemar(mc_rows, "noshard", "topo", "hit_at_1")

    payload = {
        "fit": {"n": len(fit_cases), "seed": args.fit_seed, "curve": fit_curve, "chosen_gamma": best_gamma, "beta": args.beta},
        "holdout": {"summary": summary,
                    "topo_vs_base_mcnemar": mc_topo_vs_base,
                    "topo_vs_noshard_mcnemar": mc_topo_vs_noshard},
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))

    print("\n=== HOLDOUT (frozen 50) ===", flush=True)
    print(f"  no_interaction_base (=live shardrca_full): {summary['no_interaction_base']}")
    print(f"  shardrca_topo (gamma={best_gamma}):        {summary['shardrca_topo']}")
    print(f"  no_shard (target to beat):                 {summary['no_shard']}")
    print(f"  topo vs base:    Δ t_wins={mc_topo_vs_base['treatment_only_correct']} "
          f"b_wins={mc_topo_vs_base['baseline_only_correct']} p={mc_topo_vs_base['p_value_exact']}")
    print(f"  topo vs noshard: t_wins={mc_topo_vs_noshard['treatment_only_correct']} "
          f"b_wins={mc_topo_vs_noshard['baseline_only_correct']} p={mc_topo_vs_noshard['p_value_exact']}")
    print(f"\nSaved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

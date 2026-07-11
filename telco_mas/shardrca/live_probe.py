"""Small live-LLM probe: does the live synthesizer capture what re-rankers miss?

The offline local-fusion proxy is stuck at ~0.70 Hit@1 vs a 0.93 candidate ceiling,
and four deterministic re-ranking levers all fail to move it. This probe runs the
LIVE LLM ``shardrca_full`` (synthesizer over the sharded board) against the offline
proxy on the SAME disjoint dev cases, to tell whether the live mechanism already
recovers the headroom the deterministic proxy cannot. It never touches the locked
v7 holdout or the C high-volume split.

This spends LLM budget; keep ``--n`` small. The LLM cache makes re-runs free.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ..evaluation.stats import accuracy_at_k, paired_mcnemar, reciprocal_rank
from ..llm import LLMClient, LLMError
from .fit_local_fusion import _has_trace_file, _load_excluded, runtime_case_id
from .hard_split import load_hard_cases
from .runner import run_rcaeval_case


def _run_with_retry(case, system, llm, *, attempts: int = 4):
    """Transient API connection errors must not abort the whole probe.

    The LLM cache replays any completed call for free, so retries and re-runs only
    re-issue the specific failed request.
    """
    for attempt in range(1, attempts + 1):
        try:
            return run_rcaeval_case(case, system=system, llm=llm)
        except LLMError:
            if attempt == attempts:
                raise
            time.sleep(2.0 * attempt)

PROBE_SYSTEMS = ("shardrca_full", "shardrca_local_fusion")


def run_probe(
    *,
    n: int = 15,
    seed: int = 202,
    graph_only: bool = True,
    systems=PROBE_SYSTEMS,
    exclude=("results/prereg_v7_holdout20_2026-07-03.json", "results/prereg_high_volume_draft.json"),
) -> dict:
    excluded = _load_excluded(*exclude)
    pool = [c for c in load_hard_cases(sample=None) if runtime_case_id(c.case_id) not in excluded]
    pool.sort(key=lambda c: c.case_id)
    random.Random(seed).shuffle(pool)
    if graph_only:
        pool = [c for c in pool if _has_trace_file(c)]
    cases = pool[:n]

    llm = LLMClient()
    rows = []
    for case in cases:
        rid = runtime_case_id(case.case_id)
        truth = case.ground_truth_root
        for system in systems:
            pred = _run_with_retry(case, system, llm)
            ranked = pred.ranked_roots or [pred.root]
            rows.append({
                "case_id": rid,
                "system": f"rcaeval_{system}",
                "true_root": truth,
                "predicted_root": pred.root,
                "hit_at_1": accuracy_at_k(ranked, truth, 1),
                "hit_at_3": accuracy_at_k(ranked, truth, 3),
                "mrr": reciprocal_rank(ranked, truth),
                "total_tokens": pred.total_tokens,
                "llm_calls": pred.llm_calls,
            })

    summary = {}
    for system in systems:
        key = f"rcaeval_{system}"
        srows = [r for r in rows if r["system"] == key]
        m = max(1, len(srows))
        summary[key] = {
            "n": len(srows),
            "hit_at_1": round(sum(r["hit_at_1"] for r in srows) / m, 4),
            "hit_at_3": round(sum(r["hit_at_3"] for r in srows) / m, 4),
            "mrr": round(sum(r["mrr"] for r in srows) / m, 4),
            "total_tokens": sum(r["total_tokens"] for r in srows),
        }

    treat = "rcaeval_shardrca_full"
    paired = {
        base: {m: paired_mcnemar(rows, base, treat, m) for m in ("hit_at_1", "hit_at_3")}
        for base in (f"rcaeval_{s}" for s in systems if s != "shardrca_full")
    }
    return {
        "probe": "live_synthesizer_vs_offline_proxy",
        "n_cases": len(cases),
        "graph_only": graph_only,
        "seed": seed,
        "summary": summary,
        "paired_vs_shardrca_full": paired,
        "interpretation": (
            "If live shardrca_full Hit@1 >> offline shardrca_local_fusion, the offline proxy "
            "under-estimates the mechanism and re-ranking is not the bottleneck."
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small live-LLM probe of the synthesizer vs offline proxy/oracle")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--out", default="results/live_synth_probe.json")
    args = parser.parse_args(argv)
    report = run_probe(n=args.n, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": args.out, "summary": report["summary"],
                      "paired": report["paired_vs_shardrca_full"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

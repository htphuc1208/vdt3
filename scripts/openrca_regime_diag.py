"""Decoder-free regime diagnostic for OpenRCA Telecom (option b).

Survey insight: sharding/evidence-isolation only helps when the true root is a
QUIET signal drowned by louder misleading victims (symptom amplification), NOT
when the root is the loudest concentrated signal (RCAEval regime, where a global
top-B summary retains it and a single reader wins).

This reuses the deterministic board findings already stored in the frozen
OpenRCA checkpoints (zero API cost). For every row whose task asks for the root
CAUSE COMPONENT, it measures, purely from telemetry scores:

  * root_rank         : rank of the true component among board components by score
  * loudest_is_root   : is the single highest-scoring finding the root?
  * retention single vs shard: at a fixed budget of B findings, does the true
    component survive under GLOBAL top-B vs per-shard ROUND-ROBIN allocation?

If OpenRCA roots are loud (rank 1-3, high single retention) it matches RCAEval and
sharding won't help; if they are quiet/drowned it is the regime where isolation
has a real rationale.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from statistics import mean

CKPT = "results/checkpoints/openrca_telecom_frozen/shardrca_full/row_*.json"
BUDGETS = [5, 10, 20, 40]


def _scope(ptr: str) -> str:
    p = ptr.lower()
    if "metric_node" in p:
        return "node"
    if "metric_container" in p:
        return "container"
    if "metric_service" in p or "metric_middleware" in p:
        return "service"
    if "metric_app" in p:
        return "app"
    if "trace" in p:
        return "trace"
    return "other"


def _true_component(scoring_points: str) -> str | None:
    m = re.search(r"root cause component is\s+([A-Za-z0-9_]+)", scoring_points)
    return m.group(1).strip() if m else None


def _round_robin(shard_pools: dict[str, list], B: int) -> list:
    cursors = {k: list(v) for k, v in shard_pools.items()}
    out = []
    while len(out) < B and any(cursors.values()):
        progressed = False
        for k in sorted(cursors):
            if cursors[k]:
                out.append(cursors[k].pop(0))
                progressed = True
                if len(out) >= B:
                    break
        if not progressed:
            break
    return out[:B]


def main() -> int:
    files = sorted(glob.glob(CKPT))
    rows = []
    for f in files:
        row = json.load(open(f))["row"]
        if "component" not in row["instruction"].lower():
            continue
        true_c = _true_component(row.get("scoring_points", ""))
        if not true_c:
            continue
        findings = row.get("artifacts", {}).get("board_findings", [])
        if not findings:
            continue
        # global ranked findings by score
        gl = sorted(findings, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        # per-component aggregated score (max finding score) -> component rank
        comp_score: dict[str, float] = defaultdict(float)
        for x in findings:
            comp_score[x["component"]] = max(comp_score[x["component"]], float(x.get("score", 0.0)))
        comp_ranked = [c for c, _ in sorted(comp_score.items(), key=lambda kv: kv[1], reverse=True)]
        root_rank = (comp_ranked.index(true_c) + 1) if true_c in comp_ranked else None
        n_comps = len(comp_ranked)
        loudest_is_root = bool(gl and gl[0]["component"] == true_c)
        # per-shard pools (ranked within shard)
        pools: dict[str, list] = defaultdict(list)
        for x in gl:
            pools[_scope(x.get("evidence", ""))].append(x)
        ret = {}
        for B in BUDGETS:
            single = {x["component"] for x in gl[:B]}
            shard = {x["component"] for x in _round_robin(pools, B)}
            ret[B] = (int(true_c in single), int(true_c in shard))
        rows.append({
            "row_id": row["row_id"], "true_component": true_c,
            "n_findings": len(findings), "n_components": n_comps,
            "root_rank": root_rank, "loudest_is_root": loudest_is_root,
            "retention": ret,
        })

    n = len(rows)
    print(f"component-labeled rows analyzed: {n}\n")
    ranks = [r["root_rank"] for r in rows if r["root_rank"] is not None]
    absent = sum(1 for r in rows if r["root_rank"] is None)
    print("=== is the root a LOUD or QUIET signal? ===")
    print(f"  root absent from board:            {absent}/{n}")
    print(f"  loudest finding IS the root:       {sum(r['loudest_is_root'] for r in rows)}/{n}")
    print(f"  root component rank (by score): mean={mean(ranks):.1f} median={sorted(ranks)[len(ranks)//2]} "
          f"min={min(ranks)} max={max(ranks)}  (n_components mean={mean(r['n_components'] for r in rows):.0f})")
    print(f"  root in top-1 / top-3 / top-5 components: "
          f"{sum(1 for x in ranks if x<=1)}/{sum(1 for x in ranks if x<=3)}/{sum(1 for x in ranks if x<=5)} of {len(ranks)}")
    print()
    print("=== same-budget retention: single(top-B) vs shard(round-robin) ===")
    print(f"{'B':>4} {'single':>8} {'shard':>8} {'delta':>7}")
    for B in BUDGETS:
        s = mean(r["retention"][B][0] for r in rows)
        h = mean(r["retention"][B][1] for r in rows)
        print(f"{B:>4} {s:>8.3f} {h:>8.3f} {h-s:>+7.3f}")
    json.dump({"rows": rows}, open("results/openrca_regime_diag.json", "w"), indent=2)
    print("\nSaved -> results/openrca_regime_diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

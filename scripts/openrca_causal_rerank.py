"""Feasibility test: can CROSS-LAYER topology-causal promotion recover the quiet
telecom root that anomaly-score fusion misses? (option 1, OpenRCA Telecom).

Diagnosis recap (decoder-free, from frozen checkpoints):
  * base component Hit@1 = 1/23 = 0.043; fusion collapses onto ONE loud victim
    (winner score mean 0.908), pushing 13/23 true roots out of its candidates.
  * true root is quiet: board rank mean 9.9, up to 35; roots span os/docker/db.
  * the topology used at run time was docker-only (disconnected from os_/db_ cands).

Fix under test: build the FULL cross-layer graph os->docker (trace_edges) +
docker->db (trace_span cmdb_id/dsName) + docker<->docker, widen candidates to the
board top-K (de-collapse the overconfident fusion), and re-rank by how many
symptomatic victims a candidate is graph-connected to (a common-cause signal that
can override raw anomaly score). All offline / zero API cost; component Hit@1 is
computed against the label the same way scoring does.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

CKPT = "results/checkpoints/openrca_telecom_frozen/shardrca_full/row_*.json"
PREP = "data/openrca_prepared/Telecom/rows"
TOPK = 15


def _true_component(sp: str) -> str | None:
    m = re.search(r"root cause component is\s+([A-Za-z0-9_]+)", sp)
    return m.group(1) if m else None


def _cross_layer_graph(row_id: int):
    """Undirected adjacency os<->docker<->db built from traces (both relations)."""
    adj = defaultdict(set)
    rdir = Path(PREP) / f"{row_id:03d}"
    ep = rdir / "trace_edges.parquet"
    if ep.exists():
        ed = pd.read_parquet(ep)
        for _, r in ed.iterrows():
            a, b = r.get("parent_component"), r.get("child_component")
            if a and b and str(a) != str(b):
                adj[str(a)].add(str(b)); adj[str(b)].add(str(a))
    sp = rdir / "trace" / "trace_span.parquet"
    if sp.exists():
        df = pd.read_parquet(sp, columns=["cmdb_id", "dsName"])
        pairs = df.dropna().drop_duplicates()
        for _, r in pairs.iterrows():
            a, b = str(r["cmdb_id"]), str(r["dsName"])
            if a and b and a != b:
                adj[a].add(b); adj[b].add(a)
    return adj


def _sympt_connectivity(node, symptomatic, adj):
    """How many symptomatic victims are within graph distance <=2 of node."""
    if node not in adj:
        return 0
    one = adj.get(node, set())
    two = set()
    for m in one:
        two |= adj.get(m, set())
    reach = (one | two) - {node}
    return len(reach & symptomatic)


def main() -> int:
    files = sorted(glob.glob(CKPT))
    rows = []
    for f in files:
        row = json.load(open(f))["row"]
        if "component" not in row["instruction"].lower():
            continue
        tc = _true_component(row.get("scoring_points", ""))
        if not tc:
            continue
        art = row.get("artifacts", {})
        board = art.get("board_findings", [])
        symptomatic = set(art.get("topology", {}).get("symptomatic_components", []))
        # component score (max finding) -> board-top-K candidates (de-collapse fusion)
        cs = defaultdict(float)
        for b in board:
            cs[b["component"]] = max(cs[b["component"]], float(b.get("score", 0.0)))
        ranked = [c for c, _ in sorted(cs.items(), key=lambda kv: kv[1], reverse=True)]
        cands = ranked[:TOPK]
        adj = _cross_layer_graph(row["row_id"])
        conn = {c: _sympt_connectivity(c, symptomatic - {c}, adj) for c in cands}
        base = ranked[0] if ranked else None
        # causal rule: prefer max symptomatic-connectivity, tie-break by score rank
        causal = max(cands, key=lambda c: (conn[c], -ranked.index(c))) if cands else None
        # blended rule: connectivity gates, but require candidate to be in graph
        rows.append({
            "row_id": row["row_id"], "true": tc,
            "true_in_topK": tc in cands, "true_rank": (ranked.index(tc) + 1) if tc in ranked else None,
            "base": base, "causal": causal,
            "base_hit": int(base == tc), "causal_hit": int(causal == tc),
            "true_conn": conn.get(tc, 0) if tc in conn else _sympt_connectivity(tc, symptomatic - {tc}, adj),
            "winner_conn": conn.get(causal, 0),
        })

    n = len(rows)
    base_h = sum(r["base_hit"] for r in rows)
    causal_h = sum(r["causal_hit"] for r in rows)
    topk_recall = sum(r["true_in_topK"] for r in rows)
    print(f"component rows: {n}")
    print(f"  candidate recall (true in board top-{TOPK}): {topk_recall}/{n}")
    print(f"  BASE  component Hit@1 (top anomaly score): {base_h}/{n} = {base_h/n:.3f}")
    print(f"  CAUSAL component Hit@1 (cross-layer conn): {causal_h}/{n} = {causal_h/n:.3f}")
    fixed = [r for r in rows if r["causal_hit"] and not r["base_hit"]]
    broke = [r for r in rows if r["base_hit"] and not r["causal_hit"]]
    print(f"  fixed by causal: {len(fixed)}   broke: {len(broke)}")
    for r in fixed:
        print(f"    + row{r['row_id']:>2} true={r['true']:<10} base={r['base']:<10} rank={r['true_rank']} conn={r['true_conn']}")
    # ceiling: among top-K-recall cases, how many could a perfect reranker get
    print(f"  CEILING (perfect rerank over board top-{TOPK}): {topk_recall}/{n} = {topk_recall/n:.3f}")
    Path("results/openrca_causal_rerank.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("Saved -> results/openrca_causal_rerank.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Payoff test (option 1): does an LLM causal head over DE-COLLAPSED candidates +
CROSS-LAYER telecom topology recover the quiet root on OpenRCA Telecom?

Uses the right ingredients this time (unlike the earlier RCAEval llmfuse that was
starved of graph and anchored on fusion score):
  * candidates = board top-K by anomaly score (de-collapses the overconfident
    PoE winner; candidate recall rises from 0.043 to ~0.74),
  * cross-layer graph os->docker->db built from trace_edges + trace_span,
  * each candidate annotated with layer, anomaly rank, is_symptomatic, and which
    symptomatic victims it is graph-connected to,
  * the LLM is told to pick the upstream CAUSE that explains the symptomatic
    victims, NOT the loudest victim.

Offline board/graph are free; only the final decision calls the LLM (k self-
consistency). Component Hit@1 is scored exactly like the benchmark.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from telco_mas.config import get_settings
from telco_mas.llm import LLMClient, extract_json

CKPT = "results/checkpoints/openrca_telecom_frozen/shardrca_full/row_*.json"
PREP = "data/openrca_prepared/Telecom/rows"
TOPK = 12
K = 3

SYS = """You are a telecom root-cause analyst. A failure propagates from the faulty
component through the infrastructure topology (host os_* -> container docker_* ->
service db_*). Loud/symptomatic components are usually DOWNSTREAM victims; the true
root is the component whose failure best explains the symptomatic victims it is
connected to. Do NOT just pick the component with the largest anomaly score.
Pick the root ONLY from the candidate list.
Return ONLY JSON: {"root": "component", "rationale": "short causal reason"}"""


def _true_component(sp: str) -> str | None:
    m = re.search(r"root cause component is\s+([A-Za-z0-9_]+)", sp)
    return m.group(1) if m else None


def _graph(row_id: int):
    adj = defaultdict(set)
    rdir = Path(PREP) / f"{row_id:03d}"
    ep = rdir / "trace_edges.parquet"
    if ep.exists():
        for _, r in pd.read_parquet(ep).iterrows():
            a, b = r.get("parent_component"), r.get("child_component")
            if a and b and str(a) != str(b):
                adj[str(a)].add(str(b)); adj[str(b)].add(str(a))
    sp = rdir / "trace" / "trace_span.parquet"
    if sp.exists():
        for _, r in pd.read_parquet(sp, columns=["cmdb_id", "dsName"]).dropna().drop_duplicates().iterrows():
            a, b = str(r["cmdb_id"]), str(r["dsName"])
            if a and b and a != b:
                adj[a].add(b); adj[b].add(a)
    return adj


def main() -> int:
    if not get_settings().has_api_key:
        print("need OPENAI_API_KEY"); return 2
    llm = LLMClient(cache_enabled=True)
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
        cs = defaultdict(float)
        for b in board:
            cs[b["component"]] = max(cs[b["component"]], float(b.get("score", 0.0)))
        ranked = [c for c, _ in sorted(cs.items(), key=lambda kv: kv[1], reverse=True)]
        cands = ranked[:TOPK]
        adj = _graph(row["row_id"])
        payload_cands = []
        for i, c in enumerate(cands):
            nbr = adj.get(c, set())
            sym_nbr = sorted(nbr & (symptomatic - {c}))
            payload_cands.append({
                "component": c, "layer": c.split("_")[0], "anomaly_rank": i + 1,
                "is_symptomatic": c in symptomatic,
                "connected_symptomatic_victims": sym_nbr[:8],
                "num_connected_victims": len(sym_nbr),
            })
        payload = {"candidates": payload_cands,
                   "symptomatic_victims": sorted(symptomatic)[:20],
                   "instruction": "Pick the upstream root cause component that explains the victims."}
        votes = Counter()
        for _ in range(K):
            resp = llm.chat([{"role": "system", "content": SYS},
                             {"role": "user", "content": json.dumps(payload, default=str)}], force_json=True)
            root = str(extract_json(resp.content).get("root") or "").strip()
            if root in cands:
                votes[root] += 1
        llm_pick = votes.most_common(1)[0][0] if votes else (cands[0] if cands else None)
        rows.append({"row_id": row["row_id"], "true": tc, "base": ranked[0] if ranked else None,
                     "llm": llm_pick, "true_in_topK": tc in cands,
                     "base_hit": int((ranked[0] if ranked else None) == tc), "llm_hit": int(llm_pick == tc)})
        print(f"row{row['row_id']:>2} true={tc:<10} base={rows[-1]['base']:<10} llm={llm_pick:<10} "
              f"hit={rows[-1]['llm_hit']}", flush=True)

    n = len(rows)
    print(f"\ncomponent rows: {n}")
    print(f"  candidate recall (top-{TOPK}): {sum(r['true_in_topK'] for r in rows)}/{n}")
    print(f"  BASE component Hit@1 (loudest): {sum(r['base_hit'] for r in rows)}/{n} = {sum(r['base_hit'] for r in rows)/n:.3f}")
    print(f"  LLM  component Hit@1 (causal):  {sum(r['llm_hit'] for r in rows)}/{n} = {sum(r['llm_hit'] for r in rows)/n:.3f}")
    print(f"  (frozen fusion baseline was 1/23 = 0.043)")
    Path("results/openrca_causal_llm.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("Saved -> results/openrca_causal_llm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

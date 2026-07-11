"""P0 A/B: does a HOLISTIC cross-layer causal decision head beat plain de-collapse
on OpenRCA Telecom component localization?

Compares three deterministic-board policies on the 23 cached component rows
(reuses the frozen board_findings + cross-layer graph; only the decision head
calls the LLM, k=3):
  * fusion    : the shipped collapsed log-opinion-pool winner (baseline, 0.043)
  * decollapse: board-strongest catalog component (shipped fix, ~0.130)
  * holistic  : LLM reads de-collapsed top-K candidates enriched with fault
                signals + cross-layer topology role (host->container->service),
                told to blame the upstream cause, not the loudest victim.

If holistic clearly beats decollapse it becomes the default; else de-collapse stays.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from telco_mas.config import get_settings
from telco_mas.llm import LLMClient, extract_json
from telco_mas.openrca.causal_pick import build_cross_layer_adj

CKPT = "results/checkpoints/openrca_telecom_frozen/shardrca_full/row_*.json"
PREP = "data/openrca_prepared/Telecom/rows"
TOPK = 15
K = 3

SYS = """You are a telecom root-cause decision head. Failures propagate along the
infrastructure topology host (os_*) -> container (docker_*) -> service (db_*): a
fault at an UPSTREAM element makes its downstream dependents anomalous. The
LOUDEST anomaly is usually a downstream victim, not the cause.

Given a shortlist of candidate components (each with its anomaly rank, fault
signals, and topology role relative to the anomalous set), pick the single ROOT
CAUSE. Prefer the candidate whose fault best EXPLAINS the anomalous victims it is
upstream of; do not just pick the highest anomaly rank. Choose ONLY from the
shortlist.
Return ONLY JSON: {"root": "component", "rationale": "short causal reason"}"""


def _true_component(sp: str) -> str | None:
    m = re.search(r"root cause component is\s+([A-Za-z0-9_]+)", sp)
    return m.group(1) if m else None


def _signals(board, comp, limit=3):
    items = [b for b in board if b["component"] == comp]
    items.sort(key=lambda b: float(b.get("score", 0) or 0), reverse=True)
    out = []
    for b in items[:limit]:
        s = str(b.get("signal", "")).strip()
        d = str(b.get("direction", "")).strip()
        if s:
            out.append(f"{s}{'↑' if d == 'high' else '↓' if d == 'low' else ''}")
    return out


def main() -> int:
    if not get_settings().has_api_key:
        print("need OPENAI_API_KEY"); return 2
    llm = LLMClient(cache_enabled=True)
    files = sorted(glob.glob(CKPT))
    n = fusion_h = decollapse_h = holistic_h = 0
    rows = []
    for f in files:
        row = json.load(open(f))["row"]
        if "component" not in row["instruction"].lower():
            continue
        tc = _true_component(row.get("scoring_points", ""))
        if not tc:
            continue
        n += 1
        art = row.get("artifacts", {})
        board = art.get("board_findings", [])
        allowed = set(art.get("candidate_catalog", {}).get("components", []))
        symptomatic = set(art.get("topology", {}).get("symptomatic_components", []))
        fusion_pick = art.get("fusion_candidates", [{}])[0].get("component")
        # de-collapse: board-strongest allowed component
        cs = defaultdict(float)
        for b in board:
            c = b["component"]
            if allowed and c not in allowed:
                continue
            cs[c] = max(cs[c], float(b.get("score", 0) or 0))
        ranked = [c for c, _ in sorted(cs.items(), key=lambda kv: kv[1], reverse=True)]
        decollapse_pick = ranked[0] if ranked else fusion_pick
        cands = ranked[:TOPK]
        # cross-layer graph + candidate enrichment
        adj = build_cross_layer_adj(f"{PREP}/{row['row_id']:03d}")
        cand_set = set(cands)
        payload = []
        for i, c in enumerate(cands):
            nbr = adj.get(c, set())
            sym_down = sorted((nbr & symptomatic) - {c})
            payload.append({
                "component": c, "layer": c.split("_")[0], "anomaly_rank": i + 1,
                "fault_signals": _signals(board, c),
                "anomalous_neighbors": sym_down[:6],
                "num_anomalous_neighbors": len(sym_down),
            })
        holistic_pick = decollapse_pick
        if llm is not None and len(cands) >= 2:
            votes = Counter()
            for _ in range(K):
                resp = llm.chat(
                    [{"role": "system", "content": SYS},
                     {"role": "user", "content": json.dumps(
                         {"candidates": payload, "anomalous_set": sorted(symptomatic)[:20]}, default=str)}],
                    force_json=True)
                r = str(extract_json(resp.content).get("root") or "").strip()
                if r in cand_set:
                    votes[r] += 1
            if votes:
                holistic_pick = votes.most_common(1)[0][0]
        fusion_h += int(fusion_pick == tc)
        decollapse_h += int(decollapse_pick == tc)
        holistic_h += int(holistic_pick == tc)
        rows.append({"row_id": row["row_id"], "true": tc, "fusion": fusion_pick,
                     "decollapse": decollapse_pick, "holistic": holistic_pick})
        print(f"row{row['row_id']:>2} true={tc:<10} decollapse={decollapse_pick:<10} "
              f"holistic={holistic_pick:<10} {'✓' if holistic_pick==tc else ''}", flush=True)

    print(f"\ncomponent rows: {n}")
    print(f"  fusion (shipped baseline):   {fusion_h}/{n} = {fusion_h/n:.3f}")
    print(f"  de-collapse (shipped fix):   {decollapse_h}/{n} = {decollapse_h/n:.3f}")
    print(f"  holistic causal head (P0):   {holistic_h}/{n} = {holistic_h/n:.3f}")
    Path("results/openrca_holistic_head.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("Saved -> results/openrca_holistic_head.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

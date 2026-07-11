"""P1 / Hypothesis-Evidence-Test (temporal precedence) on OpenRCA Telecom.

Every earlier attempt re-ranked candidates using the SAME coarse board evidence,
so it kept trading loud victims for demoted-but-correct loud roots. HET instead
FETCHES NEW evidence for each shortlisted hypothesis: the precise anomaly ONSET
time of that candidate from the raw per-cmdb metric series. The discriminator is
temporal precedence — a true root's anomaly starts BEFORE the victims it
propagates to. This signal is not in the board and was never tested.

Deterministic (no LLM): for each de-collapsed top-K candidate we load its metric
series (os_->node, docker_->container, db_->service/middleware), pick its dominant
KPI, detect onset (first sustained MAD-deviation from the early-window baseline),
then decide by earliest onset (optionally gated by cross-layer topology). Compared
against the shipped de-collapse and scored with the same strict metric.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import pandas as pd

from telco_mas.openrca.causal_pick import build_cross_layer_adj
from telco_mas.openrca.evaluator import evaluate_prediction

CKPT = "results/checkpoints/openrca_telecom_frozen/shardrca_full/row_*.json"
PREP = "data/openrca_prepared/Telecom/rows"
TOPK = 8
LAYER_FILE = {"os": "metric_node", "docker": "metric_container", "db": "metric_service"}


def _true_component(sp: str) -> str | None:
    m = re.search(r"root cause component is\s+([A-Za-z0-9_]+)", sp)
    return m.group(1) if m else None


def _load_layer(row_id: int, layer: str, cache: dict):
    key = (row_id, layer)
    if key in cache:
        return cache[key]
    fname = LAYER_FILE.get(layer)
    df = None
    if fname:
        for cand in ([fname, "metric_middleware"] if layer == "db" else [fname]):
            p = Path(PREP) / f"{row_id:03d}" / "metric" / f"{cand}.parquet"
            if p.exists():
                try:
                    part = pd.read_parquet(p, columns=["timestamp", "value", "cmdb_id", "name"])
                    df = part if df is None else pd.concat([df, part])
                except Exception:
                    pass
    cache[key] = df
    return df


def _onset(row_id: int, comp: str, signal: str | None, cache: dict):
    layer = comp.split("_")[0]
    df = _load_layer(row_id, layer, cache)
    if df is None:
        return None
    sub = df[df["cmdb_id"] == comp]
    if sub.empty:
        return None
    # pick the candidate's dominant KPI (board signal if available, else max variance)
    kpi = None
    if signal and (sub["name"] == signal).any():
        kpi = signal
    else:
        var = sub.groupby("name")["value"].std().sort_values(ascending=False)
        kpi = var.index[0] if len(var) else None
    if kpi is None:
        return None
    s = sub[sub["name"] == kpi].sort_values("timestamp")
    v = s["value"].astype(float).tolist()
    t = s["timestamp"].astype(float).tolist()
    if len(v) < 6:
        return None
    n0 = max(3, len(v) // 5)
    base = median(v[:n0])
    mad = median([abs(x - base) for x in v[:n0]]) or (0.05 * (abs(base) + 1e-6))
    for i in range(n0, len(v)):
        if abs(v[i] - base) > 5.0 * mad:
            return t[i]
    return None


def _dominant_signal(board, comp):
    items = sorted([b for b in board if b["component"] == comp],
                   key=lambda b: float(b.get("score", 0) or 0), reverse=True)
    return str(items[0].get("signal")) if items else None


def _het_pick(row_id, cands, board, adj):
    onsets = {}
    cache = {}
    for c in cands:
        o = _onset(row_id, c, _dominant_signal(board, c), cache)
        if o is not None:
            onsets[c] = o
    if not onsets:
        return cands[0] if cands else None, {}
    # earliest onset = precedes victims; tie-break: more upstream (more anomalous
    # descendants among candidates it points to) then anomaly rank.
    cand_set = set(onsets)
    def upstream_count(c):
        return len((adj.get(c, set()) & cand_set) - {c})
    ranked = sorted(onsets, key=lambda c: (onsets[c], -upstream_count(c), cands.index(c)))
    return ranked[0], onsets


def main() -> int:
    files = sorted(glob.glob(CKPT))
    n = decoll_h = het_h = 0
    strict = {"decollapse": 0, "het": 0}
    rows = []
    for f in files:
        row = json.load(open(f))["row"]
        is_comp = "component" in row["instruction"].lower()
        sp = row["scoring_points"]; pred = row["prediction"]
        if not is_comp:
            _, _, sc = evaluate_prediction(pred, sp)
            for k in strict:
                strict[k] += int(sc == 1.0)
            continue
        tc = _true_component(sp)
        if not tc:
            continue
        n += 1
        art = row.get("artifacts", {})
        board = art.get("board_findings", [])
        allowed = set(art.get("candidate_catalog", {}).get("components", []))
        cs = defaultdict(float)
        for b in board:
            c = b["component"]
            if allowed and c not in allowed:
                continue
            cs[c] = max(cs[c], float(b.get("score", 0) or 0))
        ranked = [c for c, _ in sorted(cs.items(), key=lambda kv: kv[1], reverse=True)]
        decoll = ranked[0] if ranked else None
        cands = ranked[:TOPK]
        adj = build_cross_layer_adj(f"{PREP}/{row['row_id']:03d}")
        het, onsets = _het_pick(row["row_id"], cands, board, adj)

        decoll_h += int(decoll == tc); het_h += int(het == tc)
        def sub(c): return re.sub(r'("root cause component":\s*")[^"]*(")', r'\g<1>' + (c or "") + r'\g<2>', pred)
        _, _, sd = evaluate_prediction(sub(decoll), sp); strict["decollapse"] += int(sd == 1.0)
        _, _, sh = evaluate_prediction(sub(het), sp); strict["het"] += int(sh == 1.0)
        rows.append({"row_id": row["row_id"], "true": tc, "decollapse": decoll, "het": het,
                     "onset_found": len(onsets)})
        mark = "✓" if het == tc else ("(decoll✓)" if decoll == tc else "")
        print(f"row{row['row_id']:>2} true={tc:<10} decoll={str(decoll):<10} het={str(het):<10} "
              f"onsets={len(onsets)}/{len(cands)} {mark}", flush=True)

    print(f"\ncomponent rows: {n}")
    print(f"  de-collapse: component {decoll_h}/{n}={decoll_h/n:.3f} | strict {strict['decollapse']}/51={strict['decollapse']/51:.3f}")
    print(f"  HET-temporal: component {het_h}/{n}={het_h/n:.3f} | strict {strict['het']}/51={strict['het']/51:.3f}")
    Path("results/openrca_het_temporal.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("Saved -> results/openrca_het_temporal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

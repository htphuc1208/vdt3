"""Cross-layer causal component picker for OpenRCA Telecom (quick-win #1).

Two fixes over the shipped path, both deterministic:

1. De-collapse candidates. The log-opinion pool collapses onto one loud victim
   (winner posterior ~0.9), pushing 13/23 true roots out of the top-8 candidates.
   We instead consider the board top-K components by anomaly score (candidate
   recall rises 0.043 -> ~0.74).

2. Cross-layer topology. The shipped graph is docker<->docker only. The full
   host->container->service graph is recoverable: trace_edges gives os<->docker,
   and trace_span links a container (cmdb_id) to the db/service it drives
   (dsName). We re-rank candidates by how many symptomatic victims each is
   graph-connected to (a common-cause signal), which can lift a quiet upstream
   root over the loudest downstream victim.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_cross_layer_adj(row_dir: str | Path) -> dict[str, set[str]]:
    """Undirected os<->docker<->db adjacency from a prepared row's trace files."""
    import pandas as pd

    row_dir = Path(row_dir)
    adj: dict[str, set[str]] = defaultdict(set)
    edges = row_dir / "trace_edges.parquet"
    if edges.exists():
        try:
            frame = pd.read_parquet(edges, columns=["parent_component", "child_component"])
            for _, r in frame.iterrows():
                a, b = r.get("parent_component"), r.get("child_component")
                if a and b and str(a) != str(b):
                    adj[str(a)].add(str(b))
                    adj[str(b)].add(str(a))
        except Exception:
            pass
    span = row_dir / "trace" / "trace_span.parquet"
    if span.exists():
        try:
            frame = pd.read_parquet(span, columns=["cmdb_id", "dsName"]).dropna().drop_duplicates()
            for _, r in frame.iterrows():
                a, b = str(r["cmdb_id"]), str(r["dsName"])
                if a and b and a != b:
                    adj[a].add(b)
                    adj[b].add(a)
        except Exception:
            pass
    return adj


def symptomatic_connectivity(node: str, symptomatic: set[str], adj: dict[str, set[str]], *, radius: int = 2) -> int:
    """Number of symptomatic victims within ``radius`` hops of ``node``."""
    if node not in adj:
        return 0
    frontier = {node}
    seen = {node}
    for _ in range(radius):
        nxt: set[str] = set()
        for m in frontier:
            nxt |= adj.get(m, set())
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return len((seen - {node}) & symptomatic)


def _component_scores(board_findings: Iterable[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for f in board_findings:
        comp = str(f.get("component") or "")
        if comp:
            scores[comp] = max(scores[comp], float(f.get("score") or 0.0))
    return scores


def causal_component_pick(
    board_findings: Iterable[dict[str, Any]],
    adj: dict[str, set[str]],
    symptomatic: set[str],
    *,
    allowed: set[str] | None = None,
    topk: int = 15,
) -> str | None:
    """Pick the root component: de-collapse to board top-K, then prefer the
    candidate connected to the most symptomatic victims (score as tie-breaker).

    Returns ``None`` when there are no board findings so the caller can fall back
    to its existing winner.
    """
    scores = _component_scores(board_findings)
    if allowed:
        scores = {c: s for c, s in scores.items() if c in allowed}
    if not scores:
        return None
    ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    cands = ranked[:topk]
    sym = {s for s in symptomatic}
    conn = {c: symptomatic_connectivity(c, sym - {c}, adj) for c in cands}
    # If the graph gives no discriminative signal at all, keep the loudest.
    if not any(conn.values()):
        return ranked[0]
    return max(cands, key=lambda c: (conn[c], -ranked.index(c)))

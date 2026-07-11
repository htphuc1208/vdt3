"""Controlled synthetic telecom RCA simulator (topology + fault propagation).

Design goals (honesty guards against circular validation):
  * The propagation model is NOT any method's decision rule. We inject a root,
    propagate along the serving topology with configurable delay + victim
    amplification + noise, and test whether each method RECOVERS the root under
    increasing realism — reporting where methods FAIL, not only where they win.
  * The two knobs that decide the regime are exposed explicitly:
      - ``victim_amp``  : how much louder downstream victims are than the root
                          (>1 makes the loudest-signal rule pick a victim).
      - ``prop_delay_s`` vs ``resolution_s`` : if propagation delay per hop is
                          below the sampling interval, onset ordering collapses and
                          temporal precedence cannot discriminate (the OpenRCA regime).
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

FAULT_FAMILIES = ("hw", "congestion", "transport", "config", "interference")


@dataclass
class Topology:
    layer: dict[str, str]
    children: dict[str, set]          # parent serves children (fault fans out down)
    parents: dict[str, set]
    nodes: list[str] = field(default_factory=list)


def build_topology(rng: random.Random, *, n_core=3, n_transport=4,
                   gnb_per_transport=4, cells_per_gnb=3) -> Topology:
    children: dict[str, set] = defaultdict(set)
    parents: dict[str, set] = defaultdict(set)
    layer: dict[str, str] = {}

    def link(a, b):
        children[a].add(b); parents[b].add(a)

    core = [f"nf_{i}" for i in range(n_core)]
    for c in core:
        layer[c] = "core"
    transport = [f"tx_{i}" for i in range(n_transport)]
    for t in transport:
        layer[t] = "transport"
        for c in rng.sample(core, k=min(len(core), rng.randint(1, 2))):
            link(c, t)
    gi = ci = 0
    for t in transport:
        for _ in range(gnb_per_transport):
            g = f"gnb_{gi}"; gi += 1; layer[g] = "gnb"; link(t, g)
            for _ in range(cells_per_gnb):
                cell = f"cell_{ci}"; ci += 1; layer[cell] = "cell"; link(g, cell)
    nodes = core + transport + [n for n in layer if n.startswith("gnb_")] + \
            [n for n in layer if n.startswith("cell_")]
    return Topology(layer=layer, children=children, parents=parents, nodes=nodes)


def descendants(topo: Topology, node: str) -> set:
    seen: set = set()
    stack = list(topo.children.get(node, ()))
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(topo.children.get(x, ()))
    return seen


def _hop_dist(topo: Topology, root: str) -> dict[str, int]:
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in topo.children.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


@dataclass
class Case:
    topo: Topology
    peak: dict[str, float]           # anomaly magnitude per alarming node
    onset: dict[str, float]          # MEASURED onset time (s) per alarming node
    symptomatic: set                 # nodes above alarm threshold
    candidates: list[str]            # alarming nodes, ranked by peak (de-collapse order)
    alarms: list[tuple[str, float]]  # (node, measured onset) alarm events
    root: str
    fault_family: str
    propagating: bool = False        # did the root fan out to >=1 downstream victim?


def generate_case(topo: Topology, rng: random.Random, *, resolution_s: float,
                  prop_delay_s: float, window_s: float = 1800.0, victim_amp: float = 1.5,
                  onset_noise_s: float | None = None, distractor_rate: float = 0.04,
                  root_peak: float = 5.0, alarm_thresh: float = 3.0) -> Case:
    # Measurement/onset-estimation noise is the realistic knob that scrambles
    # temporal ordering; on real data it is ~ the sampling interval. Temporal
    # precedence only discriminates when propagation delay exceeds this noise.
    onset_noise_s = resolution_s if onset_noise_s is None else onset_noise_s
    root = rng.choice(topo.nodes)
    fault = rng.choice(FAULT_FAMILIES)
    t0 = rng.uniform(window_s * 0.2, window_s * 0.6)
    dist = _hop_dist(topo, root)
    peak: dict[str, float] = {}
    true_onset: dict[str, float] = {}

    peak[root] = root_peak * rng.uniform(0.85, 1.15)
    true_onset[root] = t0
    victims = 0
    for n, d in dist.items():
        if d == 0:
            continue
        peak[n] = root_peak * victim_amp * rng.uniform(0.8, 1.25)     # louder downstream
        true_onset[n] = t0 + d * prop_delay_s                         # later downstream
        victims += 1
    # Loud, unrelated distractor alarms (a handful), onset anywhere in the window.
    for n in topo.nodes:
        if n in peak:
            continue
        if rng.random() < distractor_rate:
            peak[n] = root_peak * rng.uniform(0.8, 1.15)
            true_onset[n] = rng.uniform(0.0, window_s)

    # Measured onset = true onset + Gaussian estimation noise.
    onset = {n: true_onset[n] + rng.gauss(0.0, onset_noise_s) for n in peak}
    symptomatic = {n for n, p in peak.items() if p >= alarm_thresh}
    candidates = sorted(symptomatic, key=lambda n: peak[n], reverse=True)
    alarms = sorted(((n, onset[n]) for n in symptomatic), key=lambda x: x[1])
    propagating = any(n in symptomatic for n in dist if n != root)
    return Case(topo=topo, peak=peak, onset=onset, symptomatic=symptomatic,
                candidates=candidates, alarms=alarms, root=root, fault_family=fault,
                propagating=propagating)

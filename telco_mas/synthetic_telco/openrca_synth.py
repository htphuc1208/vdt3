"""Generate a label-safe synthetic RCA benchmark in the OpenRCA-Telecom schema.

Why this exists
---------------
The real OpenRCA Telecom split is only n=51 and its rows were consumed as
diagnostic evidence, so it cannot power a confirmatory claim. This generator
produces a *fresh*, arbitrarily large benchmark in the **exact OpenRCA-Telecom
on-disk schema**, so the entire existing pipeline runs unchanged:

    prepare cache -> isolated workers -> correlation-aware fusion ->
    topology/temporal causal re-rank -> falsifier -> OpenRCA evaluator ->
    preregistration + claim_audit.

Fairness contract (this is NOT a rigged benchmark)
--------------------------------------------------
* Every system (shardrca_full, single_react_sc, rca_agent_replica,
  same_board_single) reads the *same* telemetry files. Nothing is hidden from
  the single agent.
* Difficulty comes from documented real RCA properties, not from withholding
  evidence:
    1. Propagation: a moderate root fault makes its downstream callers show
       *louder* latency/error symptoms, so "pick the loudest anomaly" is wrong
       (the root-vs-dependency confusion observed on RCAEval).
    2. Scale/dispersion: a large component universe with many KPIs per
       component spreads evidence across a long context (Chain-of-Agents /
       lost-in-the-middle regime).
    3. Correlated distractors: unrelated subtrees carry spurious mild anomalies.
* The ground truth is emitted to ``record.csv`` (evaluator-only), never to the
  runtime ``query.csv``/telemetry the systems read.

The result is a benchmark where the MAS decomposition + causal re-rank has a
real mechanism, but the outcome is measured honestly, not constructed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")

# OpenRCA-Telecom reason catalog (matches telco_mas.openrca.tools.TELECOM_REASONS).
REASONS = ["CPU fault", "network delay", "network loss", "db connection limit", "db close"]

# --------------------------------------------------------------------------- #
# Fixed dependency graph (caller -> callee). A callee's fault propagates UP to
# its transitive callers, which is what the topology re-rank exploits.
#
# Tiers (edge -> mid -> backend), plus a cache tier used by services:
#   containers (docker_*)  call  services (db_*)
#   services   (db_*)      call  nodes    (os_*)   and cache (redis_*)
# --------------------------------------------------------------------------- #
CONTAINERS = [f"docker_{i:03d}" for i in range(1, 9)]     # 8 edge containers
SERVICES = [f"db_{i:03d}" for i in range(1, 13)]          # 12 mid services
NODES = [f"os_{i:03d}" for i in range(1, 23)]             # 22 backend nodes
CACHE = [f"redis_{i:03d}" for i in range(1, 11)]          # 10 cache middleware
ALL_COMPONENTS = CONTAINERS + SERVICES + NODES + CACHE

COMPONENT_FILE = {
    **{c: "metric_container.csv" for c in CONTAINERS},
    **{s: "metric_service.csv" for s in SERVICES},
    **{n: "metric_node.csv" for n in NODES},
    **{r: "metric_middleware.csv" for r in CACHE},
}
COMPONENT_LEVEL = {
    **{c: "pod" for c in CONTAINERS},
    **{s: "service" for s in SERVICES},
    **{n: "node" for n in NODES},
    **{r: "middleware" for r in CACHE},
}

# Reason -> (component pool, anomalous KPI whose *name* maps to the reason via
# telco_mas.openrca.workers._reason_hint, direction, healthy baseline, fault value).
# The KPI name is what the reason classifier keys on, so these must align.
REASON_KPI: dict[str, dict[str, Any]] = {
    "CPU fault": {
        "pools": {"node": ("CPU_util_pct", 25.0, 97.0, "high"),
                   "service": ("CPU_Used_Pct", 22.0, 96.0, "high"),
                   "pod": ("container_cpu_used", 20.0, 95.0, "high"),
                   "middleware": ("used_cpu_user", 18.0, 92.0, "high")},
    },
    "db connection limit": {
        "pools": {"service": ("Sess_Connect", 40.0, 480.0, "high"),
                   "middleware": ("blocked_clients", 0.0, 65.0, "high")},
    },
    "db close": {
        "pools": {"service": ("On_Off_State", 1.0, 0.0, "low")},
    },
    "network delay": {
        # trace-borne; also a service ping metric so metric miners see it too.
        "pools": {"service": ("tnsping_result_time", 3.0, 220.0, "high")},
    },
    "network loss": {
        "pools": {"node": ("Received_errors_packets", 0.0, 900.0, "high"),
                   "service": ("succee_rate", 100.0, 12.0, "low")},
    },
}

# Which reasons can occur on which component level (root selection universe).
# Root eligibility is restricted to component levels whose id prefix is in the
# real OpenRCA-Telecom candidate universe (os_/docker_/db_ = node/pod/service),
# so every ground-truth root is recoverable by the standard candidate catalog.
# Redis (middleware) still appears in telemetry as a cache tier and as
# correlated distractors, but is never the ground-truth root.
REASON_LEVELS = {
    "CPU fault": ["node", "service", "pod"],
    "db connection limit": ["service"],
    "db close": ["service"],
    "network delay": ["service"],
    "network loss": ["node", "service"],
}

TASK_TEMPLATES = {
    # task_index -> (requested fields, scoring template builder)
    "task_2": ("reason",),
    "task_3": ("component",),
    "task_6": ("component", "reason"),
}


@dataclass
class GeneratedCase:
    row_id: int
    task_index: str
    date_key: str
    date: datetime
    fault_start: datetime
    root_component: str
    reason: str
    downstream: list[str]
    distractors: list[str]
    instruction: str
    scoring_points: str


def _callers(callee: str, edges: list[tuple[str, str]]) -> list[str]:
    """Direct callers of a component under (caller, callee) edges."""
    return [caller for caller, target in edges if target == callee]


def _build_edges(rng: random.Random) -> list[tuple[str, str]]:
    """A fixed-ish call graph: each container calls 1-3 services; each service
    calls 1-2 nodes and 0-1 cache. Deterministic given the seed."""
    edges: list[tuple[str, str]] = []
    for c in CONTAINERS:
        for s in rng.sample(SERVICES, rng.randint(1, 3)):
            edges.append((c, s))
    for s in SERVICES:
        for n in rng.sample(NODES, rng.randint(1, 2)):
            edges.append((s, n))
        if rng.random() < 0.7:
            edges.append((s, rng.choice(CACHE)))
    return edges


def _transitive_callers(node: str, edges: list[tuple[str, str]], limit: int = 6) -> list[str]:
    seen: list[str] = []
    frontier = [node]
    guard = 0
    while frontier and guard < 50:
        guard += 1
        nxt = []
        for target in frontier:
            for caller in _callers(target, edges):
                if caller not in seen and caller != node:
                    seen.append(caller)
                    nxt.append(caller)
        frontier = nxt
    return seen[:limit]


# --------------------------------------------------------------------------- #
# Telemetry synthesis
# --------------------------------------------------------------------------- #
# Background KPIs per component level (name, healthy_mean, noise_std). These are
# emitted for every component every sample tick so the daily p05/p95 baseline is
# well defined and the anomalous KPI stands out only in the fault window.
BACKGROUND_KPIS: dict[str, list[tuple[str, float, float]]] = {
    "pod": [("container_cpu_used", 20.0, 3.0), ("container_mem_used", 45.0, 4.0),
            ("container_thread_used_pct", 30.0, 3.0)],
    "service": [("CPU_Used_Pct", 22.0, 3.0), ("MEM_Used_Pct", 55.0, 4.0),
                ("Sess_Connect", 40.0, 5.0), ("tnsping_result_time", 3.0, 0.6),
                ("On_Off_State", 1.0, 0.0), ("succee_rate", 100.0, 0.0)],
    "node": [("CPU_util_pct", 25.0, 3.0), ("Memory_used_pct", 50.0, 4.0),
             ("Received_errors_packets", 0.0, 0.0), ("Disk_io_util", 15.0, 3.0)],
    "middleware": [("used_cpu_user", 18.0, 2.5), ("used_memory", 60.0, 4.0),
                   ("blocked_clients", 0.0, 0.0), ("connected_clients", 25.0, 3.0)],
}

# Per-level file + column conventions in the OpenRCA-Telecom schema.
LONG_FILES = {  # itemid,name,bomc_id,timestamp,value,cmdb_id
    "metric_container.csv", "metric_service.csv", "metric_middleware.csv", "metric_node.csv",
}


def _emit_component_series(
    writer_rows: dict[str, list[list[Any]]],
    *,
    component: str,
    level: str,
    day_start_ms: int,
    ticks: int,
    step_ms: int,
    rng: random.Random,
    fault: dict[str, Any] | None,
) -> None:
    """Append one component's KPI samples (background + optional fault KPI)."""
    filename = COMPONENT_FILE[component]
    kpis = list(BACKGROUND_KPIS[level])
    # Overlay the fault KPI onto the background if this component is faulty/symptomatic.
    fault_kpi = None
    if fault is not None:
        fault_kpi = (fault["kpi"], fault["baseline"], fault["noise"])
        # ensure the fault KPI exists in the emitted set
        kpis = [(k, m, s) for (k, m, s) in kpis if k != fault["kpi"]] + [fault_kpi]
    for tick in range(ticks):
        ts = day_start_ms + tick * step_ms
        in_window = fault is not None and fault["start_tick"] <= tick <= fault["end_tick"]
        for name, mean, noise in kpis:
            if fault is not None and name == fault["kpi"] and in_window:
                # Abrupt step onset (not a ramp): the pre-window-half stays healthy
                # so a single agent's in-window pre/post split detects it, while the
                # fault is a minority of the full day so the daily p05/p95 baseline
                # used by the shard workers also flags it. Both detectors see it;
                # only the reasoning (root vs louder downstream symptom) differs.
                value = rng.gauss(fault["value"], fault["noise"]) if fault["noise"] > 0 else fault["value"]
            else:
                value = rng.gauss(mean, noise) if noise > 0 else mean
            row = _schema_row(filename, name=name, ts=ts, value=round(value, 4),
                              component=component)
            writer_rows[filename].append(row)


def _schema_row(filename: str, *, name: str, ts: int, value: float, component: str) -> list[Any]:
    if filename in LONG_FILES:
        # itemid,name,bomc_id,timestamp,value,cmdb_id
        itemid = abs(hash((component, name))) % 10**12
        bomc = f"ZJ-{abs(hash(component)) % 900 + 100:03d}-{abs(hash(name)) % 90 + 10:03d}"
        return [itemid, name, bomc, ts, value, component]
    raise ValueError(f"unsupported long file {filename}")


LONG_HEADER = ["itemid", "name", "bomc_id", "timestamp", "value", "cmdb_id"]
APP_HEADER = ["serviceName", "startTime", "avg_time", "num", "succee_num", "succee_rate"]
TRACE_HEADER = ["callType", "startTime", "elapsedTime", "success", "traceId", "id", "pid", "cmdb_id", "dsName", "serviceName"]


# --------------------------------------------------------------------------- #
# Case + dataset construction
# --------------------------------------------------------------------------- #
_LEVEL_POOL = {"pod": CONTAINERS, "service": SERVICES, "node": NODES, "middleware": CACHE}
# A loud propagated-symptom KPI per level (louder than the root fault -> the
# "pick the loudest anomaly" heuristic is wrong; only causal reasoning wins).
_SYMPTOM_KPI = {"pod": "container_cpu_used", "service": "tnsping_result_time",
                "node": "Disk_io_util", "middleware": "used_cpu_user"}
_SYMPTOM_BASELINE = {"container_cpu_used": (20.0, 3.0), "tnsping_result_time": (3.0, 0.6),
                     "Disk_io_util": (15.0, 3.0), "used_cpu_user": (18.0, 2.5)}


def _pick_root(rng: random.Random, reason: str) -> tuple[str, str, tuple]:
    level = rng.choice(REASON_LEVELS[reason])
    pools = REASON_KPI[reason]["pools"]
    if level not in pools:
        level = rng.choice(list(pools))
    root = rng.choice(_LEVEL_POOL[level])
    kpi, baseline, value, direction = pools[level]
    return root, level, (kpi, baseline, value, direction)


def generate_cases(n: int, seed: int, edges: list[tuple[str, str]]) -> list[GeneratedCase]:
    rng = random.Random(seed * 7919 + 13)
    task_choices = list(TASK_TEMPLATES)
    cases: list[GeneratedCase] = []
    base_date = datetime(2099, 1, 1, tzinfo=TZ)
    for i in range(n):
        task_index = task_choices[i % len(task_choices)]
        reason = REASONS[rng.randrange(len(REASONS))]
        root, level, (kpi, baseline, value, direction) = _pick_root(rng, reason)
        downstream = _transitive_callers(root, edges, limit=5)
        pool = [c for c in ALL_COMPONENTS if c != root and c not in downstream]
        distractors = rng.sample(pool, k=min(6, len(pool)))
        # Each case gets its own synthetic day so telemetry slices per case.
        date = base_date + timedelta(days=i)
        # Fault window inside a 00:00-02:00 observation day.
        fault_start = date + timedelta(minutes=45)
        instruction = _instruction(task_index, date)
        scoring = _scoring_points(task_index, root, reason, fault_start)
        cases.append(GeneratedCase(
            row_id=i, task_index=task_index, date_key=date.strftime("%Y_%m_%d"),
            date=date, fault_start=fault_start, root_component=root, reason=reason,
            downstream=downstream, distractors=distractors,
            instruction=instruction, scoring_points=scoring,
        ))
    return cases


def _instruction(task_index: str, date: datetime) -> str:
    d = date.strftime("%B %d, %Y")
    window = "00:40 to 01:10"
    if task_index == "task_2":
        ask = "identify the root cause reason for the failure during this period."
    elif task_index == "task_3":
        ask = "identify the root cause component for the failure during this period."
    else:
        ask = "identify the root cause component and the root cause reason for the failure during this period."
    return (f"On {d}, within the time range of {window}, a single failure was detected in the system. "
            f"The exact cause is unknown. Your task is to {ask}")


def _scoring_points(task_index: str, root: str, reason: str, fault_start: datetime) -> str:
    lines = []
    if task_index in ("task_3", "task_6"):
        lines.append(f"The only predicted root cause component is {root}")
    if task_index in ("task_2", "task_6"):
        lines.append(f"The only predicted root cause reason is {reason}")
    return "\n".join(lines) + "\n"


def _fault_overlay(kpi: str, baseline: float, value: float, *, start_tick: int, end_tick: int,
                   noise: float = 2.0) -> dict[str, Any]:
    return {"kpi": kpi, "baseline": baseline, "value": value, "noise": noise,
            "start_tick": start_tick, "end_tick": end_tick}


# --------------------------------------------------------------------------- #
# On-disk writer (OpenRCA-Telecom schema)
# --------------------------------------------------------------------------- #
DAY_TICKS = 240          # 1 sample/min over a 4h synthetic day
STEP_MS = 60_000         # 1 minute
# Task window is 00:40-01:10 (ticks 40-70); its pre/post midpoint is tick 55.
# Fault onsets at tick 61 (clearly in the post-half) and runs to the window end
# (tick 70): 10 faulty ticks = ~4.2% of the 240-tick day, so the daily p95 stays
# in the healthy band and the shard workers detect the fault too.
FAULT_START_TICK = 61
FAULT_END_TICK = 70
SYMPTOM_START_TICK = 61
SYMPTOM_END_TICK = 70


def _day_start_ms(date: datetime) -> int:
    return int(date.timestamp() * 1000)


def _write_case_telemetry(case: GeneratedCase, telem_dir: Path, edges: list[tuple[str, str]],
                          rng: random.Random) -> dict[str, int]:
    """Write one case's telemetry day in the OpenRCA-Telecom on-disk schema."""
    date_dir = telem_dir / case.date_key
    metric_dir = date_dir / "metric"
    trace_dir = date_dir / "trace"
    metric_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    day_ms = _day_start_ms(case.date)

    writer_rows: dict[str, list[list[Any]]] = {
        "metric_container.csv": [], "metric_service.csv": [],
        "metric_node.csv": [], "metric_middleware.csv": [],
    }

    # Root fault overlay (the true, quieter root-cause anomaly).
    root_level = COMPONENT_LEVEL[case.root_component]
    kpi, baseline, value, direction = REASON_KPI[case.reason]["pools"].get(
        root_level, next(iter(REASON_KPI[case.reason]["pools"].values())))
    root_fault = _fault_overlay(kpi, baseline, value,
                                start_tick=FAULT_START_TICK, end_tick=FAULT_END_TICK, noise=1.5)

    faults: dict[str, dict[str, Any]] = {case.root_component: root_fault}

    # Downstream propagated symptoms: LOUDER than the root (aggregation amplifies).
    # This is why "pick the loudest anomaly" is wrong and causal reasoning wins.
    for comp in case.downstream:
        lvl = COMPONENT_LEVEL[comp]
        skpi = _SYMPTOM_KPI[lvl]
        sbase, snoise = _SYMPTOM_BASELINE[skpi]
        loud = sbase + (value if value > baseline else baseline) * 3.0 + 60.0
        faults[comp] = _fault_overlay(skpi, sbase, loud,
                                      start_tick=SYMPTOM_START_TICK, end_tick=SYMPTOM_END_TICK, noise=snoise)

    # Correlated distractors: mild spurious anomalies on unrelated subtrees.
    for comp in case.distractors:
        lvl = COMPONENT_LEVEL[comp]
        skpi = _SYMPTOM_KPI[lvl]
        sbase, snoise = _SYMPTOM_BASELINE[skpi]
        mild = sbase + max(8.0, sbase * 0.8)
        faults[comp] = _fault_overlay(skpi, sbase, mild,
                                      start_tick=rng.randint(30, 80), end_tick=rng.randint(90, 120),
                                      noise=snoise)

    for comp in ALL_COMPONENTS:
        _emit_component_series(
            writer_rows, component=comp, level=COMPONENT_LEVEL[comp],
            day_start_ms=day_ms, ticks=DAY_TICKS, step_ms=STEP_MS, rng=rng,
            fault=faults.get(comp),
        )

    counts = {}
    for filename, rows in writer_rows.items():
        path = metric_dir / filename
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(LONG_HEADER)
            w.writerows(rows)
        counts[filename] = len(rows)

    # metric_app.csv (wide service KPIs) — emit osb service success rate baseline.
    app_path = metric_dir / "metric_app.csv"
    with app_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(APP_HEADER)
        for tick in range(DAY_TICKS):
            ts = day_ms + tick * STEP_MS
            w.writerow(["osb_001", ts, round(rng.gauss(0.3, 0.05), 4), 1, 1, 1.0])

    # Trace spans: build the caller->callee call graph + inflate latency on
    # edges touching the root (network-delay-style propagation).
    trace_rows = _trace_rows(case, edges, day_ms, rng)
    trace_path = trace_dir / "trace_span.csv"
    with trace_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACE_HEADER)
        w.writerows(trace_rows)
    counts["trace_span.csv"] = len(trace_rows)
    return counts


def _trace_rows(case: GeneratedCase, edges: list[tuple[str, str]], day_ms: int,
                rng: random.Random) -> list[list[Any]]:
    rows: list[list[Any]] = []
    impacted = {case.root_component, *case.downstream}
    span_seq = 0
    for tick in range(0, DAY_TICKS, 4):  # a trace burst every 4 minutes
        ts = day_ms + tick * STEP_MS
        for caller, callee in edges:
            span_seq += 1
            parent_id = f"sp{span_seq:08d}"
            span_seq += 1
            child_id = f"sp{span_seq:08d}"
            trace_id = f"tr{tick:04d}{span_seq:08d}"
            base_latency = rng.uniform(1.0, 5.0)
            in_window = FAULT_START_TICK <= tick <= FAULT_END_TICK
            if in_window and (caller in impacted or callee in impacted):
                base_latency += rng.uniform(150.0, 260.0)
            # parent span (caller service)
            rows.append(["RPC", ts, round(base_latency, 3), "True", trace_id, parent_id, "", caller, callee, ""])
            # child span (callee service), parent=parent_id
            rows.append(["RPC", ts, round(base_latency, 3), "True", trace_id, child_id, parent_id, callee, "", ""])
    return rows


def build_synth_dataset(out_dir: str | Path, *, n: int, seed: int) -> dict[str, Any]:
    out = Path(out_dir).expanduser().resolve()
    telem = out / "telemetry"
    telem.mkdir(parents=True, exist_ok=True)
    edge_rng = random.Random(seed)
    edges = _build_edges(edge_rng)
    cases = generate_cases(n, seed, edges)

    # query.csv (runtime; no labels)
    with (out / "query.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_index", "instruction", "scoring_points"])
        for c in cases:
            w.writerow([c.task_index, c.instruction, c.scoring_points])

    # record.csv (ground truth; evaluator/provenance only)
    with (out / "record.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "reason", "component", "timestamp", "datetime"])
        for c in cases:
            w.writerow([COMPONENT_LEVEL[c.root_component], c.reason, c.root_component,
                        int(c.fault_start.timestamp()),
                        c.fault_start.strftime("%Y-%m-%d %H:%M:%S")])

    telem_rng = random.Random(seed * 104729 + 7)
    total_counts: dict[str, int] = {}
    for c in cases:
        counts = _write_case_telemetry(c, telem, edges, telem_rng)
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    query_sha = hashlib.sha256((out / "query.csv").read_bytes()).hexdigest()
    manifest = {
        "dataset": "SynthTelco",
        "schema": "OpenRCA-Telecom-compatible",
        "n_cases": n,
        "seed": seed,
        "component_universe": len(ALL_COMPONENTS),
        "edges": len(edges),
        "reasons": REASONS,
        "task_mix": {t: sum(c.task_index == t for c in cases) for t in TASK_TEMPLATES},
        "query_sha256": query_sha,
        "row_counts": total_counts,
        "fairness_note": (
            "All systems read identical telemetry. Difficulty = propagation "
            "(downstream symptoms louder than the root), scale/dispersion, and "
            "correlated distractors. Ground truth is in record.csv only."
        ),
    }
    (out / "synth_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a label-safe OpenRCA-Telecom-schema synthetic RCA benchmark")
    parser.add_argument("--out", default="data/openrca/SynthTelco")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, required=True, help="declare before inspecting outcomes")
    args = parser.parse_args(argv)
    manifest = build_synth_dataset(args.out, n=args.n, seed=args.seed)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# TelcoMAS — A Multi-Agent System for Telecom Network Incident Handling

TelcoMAS is a prototype **multi-agent system** that detects, localises, explains and
remediates incidents in a mobile (5G) network. Instead of one monolithic AI agent, a
**team of specialised LLM agents** collaborates over a simulated network — triaging the
alarm storm, correlating it with operational knowledge, diagnosing the root cause from
several expert viewpoints, fusing those views by a **consensus vote**, and finally applying
and validating a fix.

> Built for the VDT2026 (DSAI) program. Every agent reasons with a live LLM through an
> **OpenAI-compatible API**, so it runs on **OpenAI** (`gpt-*`) or **DeepSeek** (`deepseek-chat`)
> with a one-line config change.

![Architecture](report/figures/architecture.png)

## Why multi-agent?

Real incident handling is a pipeline of specialised steps — detection, root-cause analysis,
data correlation, remediation and verification — and a single alarm storm usually has **one**
upstream root cause hidden behind many downstream symptoms. TelcoMAS mirrors that pipeline
with cooperating agents and shows, via a benchmark, that decomposition + consensus **localises
the true root cause more reliably** than a single agent.

The design synthesises ideas from the reference literature (see the report):

| Idea | Paper | Where in TelcoMAS |
|------|-------|-------------------|
| LLM agents for root-cause analysis | Roy et al. 2024 | the diagnosis agents |
| Tool-assisted, multi-modality observation | TAMO (Zhang et al. 2025) | the telemetry tool layer (alarms/KPIs/logs/topology) |
| Multi-agent collaboration + consensus | mABC (Zhang et al. 2024) | the weighted-vote + arbiter consensus module |
| SOP-enhanced multi-agent orchestration | Flow-of-Action (Pei et al. 2025) | the Flow-of-Action orchestrator + SOP knowledge base |
| Consensus of multi-agent systems | Zhang et al. 2026 | the confidence-weighted voting formula |

## The agent team

1. **Detection / Triage** — assess severity and the suspected domain from the alarms.
2. **Correlation** — RAG over SOP playbooks and historical incidents.
3. **Diagnosis experts** — RAN, Transport & Infrastructure, and Core specialists each propose a
   ranked root-cause hypothesis (element + fault type + confidence + evidence).
4. **Consensus** — a confidence-weighted vote over the experts, resolved by an LLM arbiter.
5. **Remediation** — turn the confirmed cause into a concrete, SOP-based plan.
6. **Validation** — apply the fix (simulated) and verify KPIs/alarms recover.

A **single-agent baseline** (one agent, all tools, no team, no consensus) is included for
comparison in the benchmark.

## Install

Requires Python 3.10+ (developed on 3.13).

```bash
# option 1: into your current environment
make install            # == pip install -r requirements.txt

# option 2: a dedicated virtualenv
make venv && source .venv/bin/activate
```

### Configure the LLM

Copy `.env.example` to `.env` and set your key:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# …or DeepSeek (OpenAI-compatible)
# OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.deepseek.com
# OPENAI_MODEL=deepseek-chat
```

## Usage

```bash
make list                       # list incident scenarios
make demo SCENARIO=fiber_cut    # run one scenario end-to-end (with agent trace)
make demo SCENARIO=dns_failure

# the single-agent baseline instead of the team:
python -m apps.cli --scenario congestion --mode single --trace

# interactive dashboard (recommended for a live demo):
make dashboard                  # http://localhost:8501

# benchmark the team vs the baseline over all scenarios (writes charts + JSON):
make bench
```

Scenarios cover RAN, Transport, Core and Power faults: `fiber_cut`, `cell_outage`,
`congestion`, `amf_misconfig`, `linecard_fault`, `site_power`, `amf_overload`, `dns_failure`,
`license`, `interference`.

## Run with Docker (one command)

```bash
make docker-build
make docker-run                 # serves the dashboard on http://localhost:8501
```

## Testing

The whole environment/tooling/pipeline core is unit-tested **without an API key** (a stubbed
LLM drives the agent loop):

```bash
make test
```

## Research / paper-grade evaluation

The research-track artifact adds public benchmark adapters and stronger scientific reporting:

```bash
# install heavier experiment dependencies
make install-research

# validate and run a label-safe RCAEval smoke/profile benchmark
make bench ARGS="--suite rcaeval --sample 30 --systems full,single,no_consensus --out results/rcaeval_sample30.json"

# staged OpenRCA integration; skips gracefully until OPENRCA_DATA_DIR/data/openrca is populated
make bench-openrca ARGS="--limit 3 --out results/openrca_smoke.json"

# ablation: isolate the contribution of RAG / consensus / arbiter (real switches, not aliases)
make bench ARGS="--systems full,single,no_rag,no_consensus,no_arbiter --runs 3 --no-cache"

# construct-validity control: hold out the exactly-matching SOP + add distractor SOPs
make bench ARGS="--systems full,single --holdout-sop --kb-distractors --no-cache"
```

Fault type is scored by **semantic family match** (fair to both systems), not exact-enum
string compliance. `--cache-only` replays cached completions offline (no live spend).

Data layout:

* `data/rcaeval` is a symlink to `/home/phucht/project/vdt2/data/rcaeval` and should contain 735 cases (RE1=375, RE2=270, RE3=90).
* `data/openrca` is a placeholder. Put OpenRCA data there or set `OPENRCA_DATA_DIR` to a directory containing `Telecom/query.csv` and `Telecom/telemetry`.
* Public-data adapters expose label-safe runtime payloads; scoring labels such as RCAEval roots and OpenRCA `scoring_points` are used only by evaluators.

The research manuscript is `report/report.md`; the original VDT-style report is preserved as
`report/report_vdt2026.md`.

## Project structure

```
telco_mas/
  environment/   synthetic 5G topology, telemetry simulator, fault scenarios (ground truth)
  knowledge/     SOP playbooks + historical incidents + TF-IDF retriever
  tools/         OpenAI function-calling tools + registry/dispatch
  agents/        detection, correlation, diagnosis, consensus, remediation, validation, orchestrator
  evaluation/    metrics, benchmark runner, charts
  llm.py         OpenAI-compatible client + tool-calling agent loop (+ cache)
  pipeline.py    run one incident (multi-agent or baseline)
  baseline.py    single-agent baseline
apps/            cli.py (rich CLI) + dashboard.py (Streamlit)
scripts/         make_architecture.py
report/          report.md + figures
tests/           unit + smoke tests
```

## Notes

* The network is **simulated** (we cannot attach to a live telecom network) but deterministic,
  so runs are reproducible; all *reasoning* is done live by the LLM agents.
* The benchmark enables an on-disk LLM cache by default so re-runs are cheap and the reported
  numbers are stable (`LLM_CACHE`, or `--no-cache`). The cache only replays real completions.

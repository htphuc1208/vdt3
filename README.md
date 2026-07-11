# ShardRCA / Telco RCA Research Artifact

This repository now focuses on **ShardRCA**, an evidence-isolated multi-agent
root-cause analysis system evaluated with preregistered, label-safe benchmark
protocols. Specialist workers inspect separate telemetry shards and a holistic
decision agent integrates their grounded findings.

## Current Evidence Status

The infrastructure-only headline result now comes from **ICASSP-SPGC 2022**, a
wireless-network fault-localization challenge built from live 5G road tests. On
the preserved 600-case held-out test set, the frozen specialist/adjudicator
system beats an equal-compute single-agent self-consistency ensemble:

| Live-5G held-out test (n=600) | Multi-agent | Single-agent | Paired result |
|---|---:|---:|---|
| Official challenge score | **0.7925** | 0.7833 | **Δ+0.00917, bootstrap 95% CI [0.00083, 0.01833]** |
| Macro-F1 | **0.7223** | 0.6957 | +0.0266 |
| Exact root-set accuracy | **0.6950** | 0.6833 | McNemar exact p=0.015625 |

Both systems use three identically configured ExtraTrees bundles. The single
agent repeats the full 336-feature view with three seeds; the multi-agent system
assigns level, temporal-dynamics, and causal-path evidence to separate
specialists, then uses an adjudicator trained only on out-of-fold training
predictions. Thresholds were fixed on a training-only validation split before
the test was evaluated. The result beats the comparable single agent, but **does
not beat** the preserved third-place score of 0.93, so there is no SOTA claim.
See [the frozen result record](results/ICAS_SPGC2022_RESULT.md) and
[the infrastructure-only benchmark audit](REAL_TELECOM_TRACK.md).

The earlier RCAEval result below is microservice evidence and is not used to
satisfy the real-telecom-network claim.

The original mechanical-fusion method was a disciplined negative result. The
promising repair is now the default: evidence-isolated specialists feed a
holistic LLM decision head, while the ineffective LLM peer-review fan-out is
removed from the primary path. On the same frozen, label-safe RCAEval-Hard
holdout:

| Evidence (RCAEval-Hard, paired n=50, prior off) | Repaired ShardRCA | Comparator | Result |
|---|---:|---|---|
| Hit@1 vs single ReAct+SC | **0.72** | 0.42 `single_react_sc` | **Δ+0.30, McNemar p=0.000729** |
| Hit@1 vs expanded-tool single | **0.72** | 0.44 `single_equal_tokens` | **Δ+0.28, p=0.001312** |
| Hit@1 vs strongest non-sharded reader | 0.72 | 0.74 `no_shard` | Δ−0.02, **p=1.0 (tie)** |
| Mean tokens/case vs single ReAct+SC | 22.0k | 17.9k | 1.23×, not strict token parity |

Honest reading:

> The repaired multi-agent pipeline significantly beats both tested single-context
> ReAct agents and statistically ties the strongest single serial reader. This is
> evidence for the repaired system, not proof that multi-agent decomposition itself
> is the cause: the gain comes from the holistic decision head, and strict token
> parity has not yet been established.

The historical mechanical method remains available as `shardrca_mechanical`:

> Mechanical fusion scored 0.54, lost to the 0.74 non-sharded reader (p=0.013),
> and its LLM peer-review round changed no decision. Those negative results are
> retained as ablations rather than hidden by the repair.

The paired result and per-case rows are in
[`results/group_a_holistic_vs_single.json`](results/group_a_holistic_vs_single.json).
The existing VDT2026 report and defense materials describe the historical method;
they should not be read as the specification of the newly promoted default.

## Architecture Overview

ShardRCA is now a specialist-and-aggregator MAS:

1. **Planner / shard builder** creates label-safe evidence shards by modality,
   component group, topology neighborhood, and time window.
2. **Specialist investigator workers** inspect only their local shard and emit
   local candidates/posteriors with evidence pointers.
3. **Holistic LLM decision head** reasons over the merged specialist evidence;
   `shardrca_mechanical` preserves the old correlation-aware fusion ablation.
4. **Causal rerankers** apply topology coverage and temporal precedence only
   when a separately preregistered validation artifact enables them; the default
   confirmatory path is no-fit/no-op.
5. **Falsifier / refinement** performs targeted top-vs-runner checks before the
   final root-cause answer.
6. **Audit artifacts** record worker findings, candidate decisions, ablations,
   usage, and label-safety provenance.

The key method ablations are `shardrca_mechanical`, `shardrca_llmboard`, and
`no_shard`; `no_interaction`, `no_refinement`, `no_topology`, and `no_falsifier`
isolate supporting stages.

## Install

```bash
make install

# Heavier public benchmark dependencies:
make install-research
```

Set an OpenAI-compatible endpoint when running live LLM benchmarks:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

## Useful Commands

```bash
# Run tests
make test

# RCAEval profile/smoke benchmark; override BENCH_ARGS for other suites.
make bench

# Fresh RCAEval confirmatory run
make bench-rcaeval-fresh

# Reproduce the frozen repaired-MAS vs single-agent comparison
make bench-rcaeval-win

# OpenRCA preparation and benchmark flow
scripts/download_openrca_telecom.sh --extract
make prepare-openrca
make prereg-openrca
make bench-openrca-full

# Rebuild readiness and claim audit reports
make readiness
make claim-audit

# Real 5G RAN benchmark
make download-icas-spgc
make bench-icas-spgc
```

## Report Direction

A research artifact with three pillars:

1. Evidence-isolated multi-agent RCA architecture with specialist workers and a
   holistic decision head.
2. Preregistered, label-safe benchmark protocol with claim auditing.
3. A significant repaired-system win over single-context ReAct, an honest tie
   against the strongest serial reader, and retained negative ablations that
   isolate the gain to the holistic decision head.


## Repository Map

```text
telco_mas/shardrca/        ShardRCA autonomous agents, interaction, fusion, falsifier, reranking
telco_mas/evaluation/      RCAEval runners, statistics, readiness, claim audit
telco_mas/openrca/         OpenRCA adapters, preparation, runner, analysis
telco_mas/icas_spgc/       Live-5G dataset guard, specialist models, paired evaluation
scripts/                  RCAEval/BARO helpers, OpenRCA download helper
```

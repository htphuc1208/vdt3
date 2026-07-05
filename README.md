# ShardRCA / Telco RCA Research Artifact

This repository now focuses on **ShardRCA**, an evidence-isolated multi-agent
root-cause analysis pipeline evaluated with preregistered, label-safe benchmark
protocols.

The Python package is still named `telco_mas` for import compatibility, but the
current research path is ShardRCA + RCAEval/OpenRCA/TelecomTS evidence, not the
legacy simulator demo.

## Current Evidence Status

The strongest current result is on RCAEval-Hard:

| Evidence | ShardRCA | Baseline | Result |
|---|---:|---:|---|
| v7 holdout Hit@1 | 0.60 | 0.20 `single_react_sc` | +0.40, exact paired p=0.007812 |
| fresh holdout Hit@1 | 0.667 | 0.375 `single_react_sc` | +0.292, exact paired p=0.039 |

Allowed claim:

> Evidence-isolated ShardRCA improves root-cause localization over a budgeted
> single-context ReAct RCA agent on preregistered, label-safe RCAEval-Hard
> holdouts.

Required caveat:

> This does not prove that multi-agent RCA beats every possible single-agent
> system. Against the global-board oracle `same_board_single`, ShardRCA ties on
> the fresh holdout.

OpenRCA Telecom is currently **diagnostic/supporting evidence only**. ShardRCA is
numerically first in the reviewed run, but the confirmatory gate is underpowered
and not claim-ready.

Primary evidence files:

- `results/positive_result_claim_package.md`
- `results/rcaeval_hard_llm_fresh_confirm24.json`
- `results/prereg_rcaeval_fresh_confirm24_frozen.json`
- `results/openrca_paired_frozen.json`
- `results/openrca_paired_frozen_analysis.json`
- `results/benchmark_readiness.json`
- `results/claim_audit_after_repair.json`

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

# OpenRCA preparation and benchmark flow
scripts/download_openrca_telecom.sh --extract
make prepare-openrca
make prereg-openrca
make bench-openrca-full

# Rebuild readiness and claim audit reports
make readiness
make claim-audit
```

## Report Direction

Write the report as a research artifact with three pillars:

1. Evidence-isolated multi-agent RCA architecture.
2. Preregistered, label-safe benchmark protocol with claim auditing.
3. Replicated RCAEval-Hard win over budgeted single-context ReAct, plus clear
   boundaries on oracle-like single agents and real-telecom evidence.

Avoid phrasing the current result as a real-telecom SOTA win. The next full-claim
target is a clean TN-RCA530/OpenRCA/TeleLogsAgent confirmatory benchmark with
non-contaminated rows.

## Repository Map

```text
telco_mas/shardrca/        ShardRCA miners, board, fusion, falsifier, reranking
telco_mas/evaluation/      RCAEval runners, statistics, readiness, claim audit
telco_mas/openrca/         OpenRCA adapters, preparation, runner, analysis
telco_mas/telecomts/       TelecomTS adapter and diagnostics
telco_mas/telelogs_agent/  TeleLogsAgent adapter/tool-mode scaffolding
results/                  Local benchmark artifacts and preregistrations
scripts/                  Dataset download helpers
```

Legacy simulator modules may still exist where required by synthetic fallback and
readiness checks, but they are no longer the advertised product/demo surface.

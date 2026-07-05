# Post-Review Full-Claim Recovery Strategy

The reviewer feedback changes the strategy from "lead with the clean positive
result" to "recover a full claim only after fixing measurable failure modes."
The target remains a full claim, but the current artifacts are diagnostic or
supporting evidence, not final proof.

## Current Evidence Status

### RCAEval-Hard v7: Qualified Supporting Result Only

ShardRCA has one statistically significant result:

- Artifact: `results/rcaeval_hard_llm_v7_holdout20.json`
- Preregistration: `results/prereg_v7_holdout20_2026-07-03.json`
- Budgeted single baseline analysis: `results/rcaeval_hard_llm_v7_holdout20_analysis_sc.json`
- ShardRCA Hit@1: 0.60
- `single_react_sc` Hit@1: 0.20
- Paired exact McNemar p: 0.007812

Allowed wording:

> On a preregistered, label-safe RCAEval-Hard microservice holdout, ShardRCA
> improves root localization over a budgeted single-context ReAct baseline.

Mandatory limitations:

- This is microservice RCA, not telecom alarm RCA.
- n=20 only supports large-effect diagnostics.
- `same_board_single` reaches Hit@1 0.70, above ShardRCA's 0.60, so this does
  not prove a mechanism-level win over a compact-board single oracle.
- Fault-type accuracy is not improved.

### OpenRCA Telecom: Failed Confirmatory Gate

The frozen OpenRCA Telecom run is real-operational evidence, but it does not
support a MAS-win claim.

- Artifact: `results/openrca_paired_frozen.json`
- Analysis: `results/openrca_paired_frozen_analysis.json`
- Confirmatory rows after contamination ledger: 50
- ShardRCA strict: 0.10
- `single_react_sc` strict: 0.04
- `rca_agent_replica` strict: 0.04
- `same_board_single` strict: 0.04
- Holm-adjusted confirmatory p: 0.5
- High-volume bin: all systems 0% strict

Allowed wording:

> OpenRCA Telecom exposed low absolute accuracy and an underpowered positive
> direction; it is diagnostic evidence, not a clear real-operational win.

### TelecomTS: Failed Synthetic Fallback Gate

- Artifact: `results/telecomts_test_frozen_paired.json`
- Analysis: `results/telecomts_test_frozen_paired_analysis.json`
- ShardRCA macro: 0.14
- `same_board_single` macro: 0.1367
- Paired exact p: 1.0

Allowed wording:

> TelecomTS currently behaves like a development diagnostic: ShardRCA is near
> random-chance and does not beat same-board or equal-call baselines.

## Recovery Order

1. Run OpenRCA error taxonomy:

   ```bash
   python3 -m telco_mas.openrca.error_analysis \
     results/openrca_paired_frozen.json \
     --out results/openrca_error_taxonomy.json
   ```

2. Run the no-LLM `heuristic_floor` on consumed OpenRCA rows to determine
   whether the current LLM systems are below a simple telemetry floor.

3. Repair the algorithm only on declared development/validation rows:

   - runtime-derived component candidate catalogs;
   - protocol-prior reason catalog with explicit provenance;
   - targeted falsifier with support/refute evidence;
   - topology and temporal reranking from prepared trace edges and onsets;
   - bounded iterative top-vs-runner refinement.

4. Fit OpenRCA repair weights only on declared validation rows:

   ```bash
   python3 -m telco_mas.openrca.fit_repair_weights \
     results/openrca_paired_frozen.json \
     --dev-rows <comma-separated-validation-row-ids> \
     --forbid-prereg results/prereg_openrca_telecom_frozen.json \
     --out-weights results/weights/openrca_repair_v1.json \
     --out-report results/openrca_repair_fit_report.json
   ```

5. Freeze a new confirmatory benchmark/run only after code, prompts, weights,
   systems, row IDs, and contamination ledger are fixed.

## Full-Claim Gate

A full MAS-over-single claim is allowed only if a new frozen run passes
`claim_audit --strict` and satisfies all of the following:

- ShardRCA beats both `single_react_sc` and `rca_agent_replica` or the official
  benchmark single-agent baseline.
- Effect is at least +0.10 absolute strict accuracy or at least 20% relative
  error reduction.
- Holm-adjusted exact paired p <= 0.05 for confirmatory comparisons.
- Same-board oracle gap is <= 0.05, or the paper explicitly limits the claim to
  operational budgeted single agents.
- High-volume or hard-bin delta is positive.
- Ablations `no_falsifier`, `no_topology`, and `no_refinement` are reported.
- Candidate catalog provenance is not label-derived.

## Benchmark Priority

1. **TN-RCA530 / TN-AutoRCA**: best full-claim target if official files are
   obtained; it is real telecom alarm RCA with graph/alarm inputs and enough
   scale for a confirmatory claim.
2. **TeleLogsAgent official tool mode**: best gated fallback for synthetic 5G
   agentic tool-use RCA after access is authorized.
3. **OpenRCA 2.0**: future operational RCA target if official artifacts become
   available.
4. **OpenRCA 1.0 Telecom**: now a consumed diagnostic/development path unless a
   fresh, properly frozen split or external extension is obtained.
5. **RCAEval-Hard/high-volume**: supporting microservice evidence only.

## Paper Positioning Until a New Win Exists

The honest interim position is a system paper with a strong methodology
contribution:

> ShardRCA is an evidence-isolated multi-agent RCA system with strict
> preregistration, label-safety, baseline diversity, and automated claim audit.
> Current positive evidence is limited to budgeted microservice RCA; current
> telecom/operational runs expose failure modes that motivate targeted repair.

Do not write a title, abstract, or headline table implying that ShardRCA already
beats strong single agents on telecom RCA.

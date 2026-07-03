# TelcoMAS: Tool-Grounded Multi-Agent Root-Cause Analysis with Calibrated Consensus

**Artifact status.** This manuscript is the research-track version of the
TelcoMAS report. The original short VDT report is preserved as
`report/report_vdt2026.md`.

## Abstract

Root-cause analysis (RCA) in telecom and cloud operations requires reasoning over
alarm storms, time-varying KPIs, logs, topology, and operational playbooks. We
present **TelcoMAS**, a tool-grounded multi-agent RCA prototype that decomposes
incident handling into triage, knowledge correlation, domain-specialist
diagnosis, calibrated consensus, remediation, and validation. The system is
motivated by ReAct-style reasoning/action loops, recent LLM-agent RCA studies,
SOP-guided multi-agent workflows, and tool-assisted multimodal RCA. Its main
engineering contribution is an auditable, mABC-inspired calibrated
evidence-weighted consensus heuristic over domain experts, with an arbiter used
only when the vote margin is low. We evaluate TelcoMAS on a deterministic 5G
incident simulator and add adapters for public RCA benchmarks: RCAEval and
OpenRCA. Current artifact tests pass end-to-end, RCAEval is linked and validated
at 735 cases, and OpenRCA is integrated in a staged manner for resource-aware
future runs.

## 1. Introduction

Operational RCA is difficult because one upstream fault can create many
downstream symptoms. In telecom networks, a fiber cut, site power failure, radio
interference event, or core-network misconfiguration can trigger alarms across
transport, RAN, and core elements. In cloud-native systems, similar propagation
appears across services, containers, traces, metrics, and logs. LLMs are useful
for summarizing such evidence, but raw prompting is brittle: large telemetry
contexts can exceed context windows, hallucinated actions can derail
investigations, and self-reported confidence is often poorly calibrated.

TelcoMAS takes the position that RCA agents should not be monolithic chatbots.
Instead, they should interact with bounded tools, specialize by operational
role, expose evidence, and fuse hypotheses with calibrated confidence. This is
consistent with ReAct's reasoning/action paradigm, RCA-agent style telemetry
retrieval in OpenRCA, TAMO's tool-assisted multimodal observation, Flow-of-Action
SOP guidance, and mABC's multi-agent voting.

## 2. Method

### 2.1 Workflow

The pipeline is:

1. **Detection/triage**: summarize alarms, severity, affected elements, and the
   suspected domain.
2. **Knowledge correlation**: retrieve SOPs and historical incidents using a
   lightweight TF-IDF retriever.
3. **Parallel expert diagnosis**: RAN, Transport/Infrastructure, and Core
   experts independently inspect alarms, KPIs, logs, topology, diagnostics, and
   knowledge-base evidence.
4. **Calibrated consensus**: hypotheses are scored with calibrated confidence,
   evidence support, topology support, and RAG/SOP support. Agreement among
   distinct experts adds a small bonus. The LLM arbiter is called only when the
   top candidates are close or experts conflict.
5. **Remediation**: the confirmed root cause is mapped to an SOP-grounded action.
6. **Validation**: the fix is applied to the simulator and resolution is judged
   from actual post-action environment state, not from the model's claim.

### 2.2 Calibrated Evidence-Weighted Consensus

The previous prototype used raw confidence-weighted voting. The current method
shrinks raw LLM confidence toward 0.5 and adds bounded evidence terms:

```
score(e) = sum_i [
  calibrate(conf_i)
  + evidence_bonus_i
  + topology_bonus_i
  + rag_bonus_i
] * 1[expert_i predicts e]
+ beta * (distinct_experts_on_e - 1)
```

This is deliberately presented as an **engineering heuristic**, not as a new
control-theoretic consensus proof. The calibration motivation follows work on
confidence calibration, and the risk/coverage framing follows conformal-style
abstention ideas. The arbiter remains available for low-margin conflicts, but
the common high-margin path is numeric and auditable.

### 2.3 External Benchmark Adapters

We added a common `ExternalBenchmarkCase` abstraction with a label-safe
`inference_payload()`. Ground-truth labels remain available only to scorers.

**RCAEval.** `data/rcaeval` is a symlink to the downloaded RCAEval data in
`/home/phucht/project/vdt2/data/rcaeval`. The adapter validates the official
case counts:

| Suite | Cases |
|---|---:|
| RE1 | 375 |
| RE2 | 270 |
| RE3 | 90 |
| Total | 735 |

**OpenRCA.** The OpenRCA integration expects `OPENRCA_DATA_DIR` or
`data/openrca` to contain `Telecom/query.csv` and `Telecom/telemetry`. Runtime
tasks expose only task id and instruction; `scoring_points` are used only after
prediction generation. Because the current machine has limited RAM and the
OpenRCA data is not present locally, the CLI skips gracefully unless
`--strict-data` is used.

## 3. Experimental Setup

### 3.1 Telco v1 Simulator

The telecom simulator contains ten deterministic scenarios: fiber cut, cell
outage, congestion, AMF misconfiguration, line-card fault, site power loss, AMF
overload, DNS failure, license exhaustion, and interference. Each scenario has a
single injected root cause, downstream symptom propagation, KPIs, alarms, logs,
diagnostics, and a remediation signature.

### 3.2 RCAEval Smoke/Profile Benchmark

The current RCAEval runner is a label-safe profile benchmark. It summarizes
pre/post metric shifts, ranks likely root services, and reports Hit@k, MRR,
fault accuracy, and bootstrap 95% confidence intervals. It is not yet the final
LLM-agent RCAEval result; it is the artifact bridge that proves data ingestion,
label separation, scoring, and confidence interval reporting.

Command used:

```bash
python3 -m telco_mas.evaluation.run_benchmark \
  --suite rcaeval \
  --sample 30 \
  --systems full,single,no_consensus \
  --out results/rcaeval_sample30.json
```

### 3.3 OpenRCA Staged Integration

Command used:

```bash
python3 -m telco_mas.openrca.cli --limit 3 --out results/openrca_smoke.json
```

Current status: skipped, because `data/openrca/Telecom` is not populated. Once
the dataset is placed there, the same CLI evaluates predictions with the
OpenRCA scoring format.

### 3.4 Ablation protocol (real component switches)

To isolate the contribution of each component, the pipeline exposes real ablation
switches (`PipelineConfig` in `agents/orchestrator.py`), *not* aliases of the full
system:

- **`no_rag`** — skip the correlation agent and remove the knowledge-base tools from
  the diagnosis experts (isolates the value of RAG for localization).
- **`no_consensus`** — skip the consensus module and take the single most confident
  expert (isolates the fusion mechanism vs. best-expert).
- **`no_arbiter`** — run the numeric weighted vote but never call the LLM arbiter
  (isolates the arbiter's marginal value over the numeric vote).
- **`no_partition`** — give every expert unrestricted deep telemetry access
  (isolates the value of information diversity by domain partition).
- **`no_debate`** — skip the cross-examination round before consensus
  (isolates the value of one disagreement-resolution round).

Each switch changes the executed pipeline (unit-tested in `tests/test_p0_hardening.py`).
The ablation table (§4.3) is produced by a live run and is saved separately by
default:

```bash
python3 -m telco_mas.evaluation.run_benchmark \
  --systems full,single,no_rag,no_consensus,no_arbiter,no_partition,no_debate --runs 3 --no-cache
# -> results/ablation_telco_v1_runs3.json
```

### 3.5 Construct-validity controls (hold-out + distractors)

Because the SOP/incident knowledge base is generated from the same fault library that
generates incidents, naive RAG can "read the answer key". Two controls address this:

- **`--holdout-sop`** removes each scenario's exactly-matching SOP *and* same-fault
  historical incidents from the retriever, forcing the system to generalize from
  telemetry instead of retrieving the solution.
- **`--kb-distractors`** injects plausible off-target SOPs/incidents (handover failure,
  GNSS loss, BGP flap, cert expiry, …) so retrieval is non-trivial.

```bash
python3 -m telco_mas.evaluation.run_benchmark --systems full,single --holdout-sop --kb-distractors --no-cache
# -> results/construct_holdout_distractors_telco_v1.json
```

### 3.6 Telco v2 stress suite

The original `telco_v1` suite has only ten deterministic single-root scenarios. To
avoid ceiling-effect claims, the artifact now includes `telco_v2`: 60 generated
synthetic scenarios balanced across the ten fault families. Each case carries stress
metadata such as `rag_required`, `expert_disagreement`, `arbiter_required`,
`missing_noisy_telemetry`, `multi_fault`, `distractor_alarms`, and `no_exact_sop`.
Several tags are active in the simulator/runner: multi-fault cases inject a secondary
fault, noisy cases perturb the initial alarm view, distractor cases add irrelevant
alarms, and `no_exact_sop` cases hold out the matching SOP during retrieval.

```bash
python3 -m telco_mas.evaluation.run_benchmark \
  --suite telco_v2 \
  --systems full,single,no_rag,no_consensus,no_arbiter \
  --runs 3 --no-cache
```

### 3.7 Telco v3 hard-regime hypothesis test

`telco_v3` is the suite intended for the conditional multi-agent claim. It uses a
larger topology, cross-domain masquerade faults, multi-fault cases, and false-alarm
noise. The method under test is also stronger than the original prototype: experts
can deep-inspect only their own domains, a cross-examination round runs when experts
disagree, and consensus weights verified diagnostics plus topology coverage rather
than keyword matches in an expert's prose.

The intended scientific claim is conditional: if results support it, we can say that
TelcoMAS improves strict diagnosis or end-to-end primary-fault resolution over a
strong unrestricted single-agent baseline on hard, information-partitioned RCA cases.
We should not claim that multi-agent systems are universally superior.

```bash
python3 -m telco_mas.evaluation.run_benchmark \
  --suite telco_v3 \
  --systems full,single,no_rag,no_consensus,no_arbiter,no_partition,no_debate \
  --runs 3 --no-cache \
  --out results/telco_v3_ablation_runs3.json
```

## 4. Results

### 4.1 Telco v1 Cached Benchmark (fair metric)

The existing cached live-LLM benchmark in `results/benchmark.json` reports:

| Metric | Multi-agent | Single-agent |
|---|---:|---:|
| Localization accuracy | 100% | 90% |
| Root-cause keyword accuracy | 100% | 100% |
| Fault-type family accuracy (semantic) | 90% | 90% |
| Diagnosis accuracy | 100% | 90% |
| Resolution rate | 100% | 90% |
| Avg tokens / incident | 24,190 | 7,434 |
| Avg latency / incident | 48.4s | 8.6s |

**Metric correction (important).** An earlier version scored fault type by *exact*
match to the canonical enum. Because only the multi-agent experts were prompted with
the enum, the single agent — which emitted equally valid alarm-style labels such as
`MAINS_FAIL`, `HIGH_CPU`, `CONFIG_MISMATCH` — scored 0% purely as a label-space
artifact, not a capability gap. We replaced this with a **semantic family match**
(`fault_type_match` in `evaluation/metrics.py`, applied identically to both systems).
Under the fair metric, fault-type accuracy is **tied at 90%**.

**Honest interpretation.** After removing that artifact, the genuine multi-agent
advantage on this suite is **one case each** in localization, diagnosis, and
resolution (10/10 vs 9/10, n=10) at roughly **3.25× the token cost and 5.6× the
latency**. With n=10 and a single discordant pair, a paired McNemar test is not
significant (p≈1.0). The result supports feasibility of the pipeline; it is **not**
evidence that multi-agent beats single-agent — the synthetic suite has a ceiling
(the single agent already scores ~90–100%), so the comparison is under-powered by
construction. Stronger claims require the harder scenarios, ablations, and external
benchmarks described below.

### 4.2 RCAEval Sample-30 Profile Smoke

On a stratified sample of 30 RCAEval cases:

| System label | Hit@1 | Hit@3 | MRR | Fault accuracy |
|---|---:|---:|---:|---:|
| rcaeval_full | 0.667 [0.500, 0.833] | 0.767 [0.600, 0.900] | 0.742 [0.606, 0.868] | 0.467 [0.300, 0.633] |
| rcaeval_single | 0.667 [0.500, 0.833] | 0.767 [0.600, 0.900] | 0.742 [0.606, 0.868] | 0.467 [0.300, 0.633] |
| rcaeval_no_consensus | 0.667 [0.500, 0.833] | 0.767 [0.600, 0.900] | 0.742 [0.606, 0.868] | 0.467 [0.300, 0.633] |

These rows are intentionally identical because the current RCAEval path is a
profile-smoke scorer shared across labels. The purpose of this table is to show
that public-data ingestion, scoring, and CI reporting are working. The next
paper-grade experiment should replace these smoke systems with true LLM-agent
and ablation runs.

### 4.3 Ablation and construct-validity controls

The July 2026 `telco_v1` ablation run under the earlier loose diagnosis metric shows
a clear ceiling effect:

| System | Localization | Root cause keyword | Loose diagnosis | Resolution | Avg tokens | Avg tool calls | Avg LLM calls | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100% | 100% | 100% | 100% | 24,522 | 30 | 19 | 29.0s |
| no_rag | 100% | 100% | 100% | 100% | 25,051 | 34 | 21 | 26.6s |
| no_consensus | 100% | 100% | 100% | 100% | 22,942 | 30 | 18 | 28.6s |
| no_arbiter | 100% | 100% | 100% | 100% | 23,382 | 30 | 18 | 29.0s |
| single | 90% | 100% | 90% | 63% | 7,226 | 6 | 6 | 8.3s |

This table supports only a feasibility claim. Because `full`, `no_rag`,
`no_consensus`, and `no_arbiter` are tied at 100%, `telco_v1` does not isolate the
causal contribution of RAG, consensus, or the arbiter. It shows that the scenarios are
too easy for the multi-agent variants and that the single-agent gap is concentrated in
end-to-end remediation rather than root-cause wording.

The construct-validity control (`--holdout-sop --kb-distractors`) is more informative:

| System | Localization | Root cause keyword | Loose diagnosis | Resolution | Avg tokens | Avg tool calls | Avg LLM calls | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 100% | 100% | 100% | 70% | 26,037 | 34 | 20 | 30.4s |
| single | 90% | 100% | 90% | 60% | 7,632 | 6 | 7 | 9.1s |

Holding out the exact SOP and adding distractors leaves diagnosis near the ceiling but
reduces resolution, exposing a remediation/action-execution fragility. This is exactly
why the benchmark now reports strict diagnosis, remediation target correctness,
action/SOP correctness, simulator-grounded resolution, and solved-cases-per-10k-tokens.
Future reported tables should use the new strict metric: correct element + semantic
fault family + causal explanation. Repeated LLM runs must be treated as stochastic
variance over the same scenarios, not as new independent samples.

The next publishable result should come from `telco_v3`, with `telco_v1` retained as
a ceiling-effect feasibility control and `telco_v2` as broader synthetic coverage.
The ablation reading is mechanistic: `full > no_partition` supports information
diversity, `full > no_debate` supports cross-examination, and `full > no_consensus`
supports fusion beyond selecting the most confident expert. If `full` does not beat
the strong single-agent baseline on `telco_v3`, the correct conclusion is that this
mechanism or suite does not yet establish the advantage.

## 5. Threats to Validity

**Synthetic-to-real gap and KB circularity.** Telco v1 is deterministic and generated
from a fault library that also informs the knowledge base, so naive RAG can retrieve
the answer. The `--holdout-sop` and `--kb-distractors` controls (§3.5) mitigate this by
removing the exact answer and adding distractors, but the environment remains synthetic;
it is useful for controlled debugging, not for broad field claims.

**OpenRCA resources.** OpenRCA is the strongest LLM-agent external benchmark,
but it requires large telemetry storage and careful chunked processing. It is
integrated but not yet locally populated.

**RCAEval domain shift.** RCAEval is microservice-focused, while TelcoMAS is
telecom-focused. Positive RCAEval results would show transfer of the reasoning
pattern, not telecom-specific field validation.

**LLM variance and calibration.** LLM outputs can vary by provider, model,
temperature, and prompt. Reported paper results must include exact model,
provider, temperature, cache state, run count, and confidence intervals.

**Remediation safety.** The current remediation tool acts only on a simulator.
Production use would require approval gates, rollback plans, RBAC, audit logs,
and prompt-injection controls.

## 6. Artifact Status

Implemented in this revision:

- RCAEval symlink and label-safe adapter; OpenRCA staged integration.
- `ExternalBenchmarkCase` abstraction; extended benchmark CLI (suites, systems, sampling, runs, seeds).
- Verifiable-evidence-weighted consensus; partitioned parallel expert diagnosis; validation grounded in simulator state.

Scientific-hardening (P0) changes in this revision:

- **Fair fault-type metric** — semantic family match replacing exact-enum (removed the
  90%-vs-0% label-space artifact; both systems now 90%).
- **Strict diagnosis metric** — headline diagnosis now requires correct element,
  semantic fault family, and causal explanation; loose keyword matching remains a
  secondary metric.
- **Remediation decomposition** — target/action/SOP correctness, validation evidence,
  simulator-grounded resolution, and solved-cases-per-10k-tokens are reported separately.
- **Paper-grade intervals/tests** — Wilson intervals for binary rates, exact paired
  McNemar tests, and scenario-level paired bootstrap effects.
- **Real ablation switches** — `no_rag` / `no_consensus` / `no_arbiter` change the executed
  pipeline (previously they were aliases of the full system).
- **Construct-validity controls** — `--holdout-sop` and `--kb-distractors`.
- **Telco v2 stress suite** — 60 generated scenarios balanced over the ten fault families,
  with active stress tags for multi-fault, noisy, distractor, and no-exact-SOP cases.
- **Telco v3 hard suite** — larger topology, cross-domain masquerade faults, multi-fault
  cases, false alarms, partition/debate ablations, and primary-fault-cleared scoring.
- **External benchmark separation** — RCAEval profile mode is explicitly marked as a smoke
  test; RCAEval/OpenRCA CLIs expose live LLM modes for label-safe external predictions.
- **`--cache-only`** offline guard so results can be regenerated without live spend.

Verification:

```bash
python3 -m pytest -q
# 56 passed
```

## References

1. Yao et al. ReAct: Synergizing Reasoning and Acting in Language Models.
   https://arxiv.org/abs/2210.03629
2. Roy et al. Exploring LLM-based Agents for Root Cause Analysis.
   https://arxiv.org/abs/2403.04123
3. Zhang et al. mABC: Multi-Agent Blockchain-inspired Collaboration for Root
   Cause Analysis in Micro-Services Architecture.
   https://aclanthology.org/2024.findings-emnlp.232/
4. Pei et al. Flow-of-Action: SOP Enhanced LLM-Based Multi-Agent System for
   Root Cause Analysis. https://arxiv.org/abs/2502.08224
5. Zhang et al. TAMO: Fine-Grained Root Cause Analysis via Tool-Assisted LLM
   Agent with Multi-Modality Observation Data. https://arxiv.org/abs/2504.20462
6. Xu et al. OpenRCA: Can Large Language Models Locate the Root Cause of
   Software Failures? https://github.com/microsoft/OpenRCA
7. Pham et al. RCAEval: A Benchmark for Root Cause Analysis of Microservice
   Systems with Telemetry Data. https://github.com/phamquiluan/RCAEval
8. Guo et al. On Calibration of Modern Neural Networks.
   https://arxiv.org/abs/1706.04599
9. Shafer and Vovk. A Tutorial on Conformal Prediction.
   https://arxiv.org/abs/0706.3188
10. Simba: Root Cause Analysis of Anomalies in 5G RAN Using Graph Neural
    Network and Transformer. https://arxiv.org/html/2406.15638v1

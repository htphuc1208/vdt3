# PLAN v5 -- Make MAS Beat Single Agent Clearly, Without Benchmark Rigging

**Status:** superseded as the headline protocol by `PLAN_V6_REAL_TELCO_PROTOCOL.md`; v7 label-safe
RCAEval-Hard is supporting evidence only.

**Objective:** produce a defensible MAS system that beats a strong single-agent baseline by a clear
margin on real RCA data.
**Current evidence:** the pre-registered v7 RCAEval-Hard holdout
(`results/prereg_v7_holdout20_2026-07-03.json`,
`results/rcaeval_hard_llm_v7_holdout20.json`) passes the primary clear-win gate against the
budgeted single-context self-consistency baseline: `shardrca_full` reaches **Hit@1 = 0.60** vs
`single_react_sc` **0.20**, paired mean difference **+0.40**, exact McNemar **p = 0.007812**.
This supports the narrower claim "higher accuracy than budgeted single-context RCA at higher
parallel compute." It does **not** yet support the stronger claim that MAS beats a global-board
single oracle: `same_board_single` reaches **Hit@1 = 0.70** on the same holdout.

**Important update:** per the real-benchmark-first objective, this file is now an appendix/engineering
log. The final claim must follow `PLAN_V6_REAL_TELCO_PROTOCOL.md`: OpenRCA Telecom or TN-RCA530 first;
synthetic-only evidence requires a telecom-valid dataset design and explicit limitations.

---

## 1. Cold Audit: Why v4 Did Not Win

### 1.1 Result

RCAEval live run, n=50:

| System | Hit@1 | Hit@3 | MRR | Fault acc | Total tokens |
|---|---:|---:|---:|---:|---:|
| `rcaeval_shardrca_full` | 0.94 | 0.96 | 0.9507 | 0.64 | 839,725 |
| `rcaeval_single` | 0.94 | 0.96 | 0.9507 | 0.64 | 280,034 |
| `rcaeval_single_sc` | 0.94 | 0.96 | 0.9507 | 0.64 | 840,005 |

Paired comparison against `single_sc`:

| Metric | MAS-only correct | single_sc-only correct | Mean diff | p-value |
|---|---:|---:|---:|---:|
| Hit@1 | 0 | 0 | 0.0 | 1.0 |
| Hit@3 | 0 | 0 | 0.0 | 1.0 |
| fault acc | 0 | 0 | 0.0 | 1.0 |

The root predictions are identical case-by-case between `shardrca_full` and `single_sc`.

### 1.2 Diagnosis

The current implementation accidentally gives the single baseline the MAS advantage:

1. `single_baseline.py` runs the same deterministic miners over all metrics/logs/traces.
2. It builds the same compact blackboard as MAS.
3. It then asks the same synthesizer to choose a root cause.

So the comparison is **not** "parallel context MAS vs one-context single"; it is "same evidence
board + different wrapper." Unsurprisingly, both systems tie.

There was also a more serious label-safety bug in the RCAEval adapter: `case_id` exposed strings
such as `RCAEval-RE1-OB-adservice_cpu-1`, and `tags` included the fault type. Therefore the
existing RCAEval LLM numbers are useful only as pipeline smoke tests, not as scientific evidence.
The adapter must expose only opaque runtime IDs during inference.

After this audit, a second label-safety bug was found: telemetry catalog summaries exposed a
filesystem `root` path containing label-derived directory names such as `payment_mem/2`. All live
v6 LLM numbers are therefore invalid as scientific evidence. The v7 checkpoint namespace was created
after removing that path from model-facing summaries and adding prompt leak tests.

### 1.3 Dataset issue

RCAEval random n=50 is too easy for root localization:

- RE1/RE2 root localization is near ceiling.
- RE3 uses code-level labels `F1..F5`; our current fault taxonomy (`cpu/mem/disk/delay/loss/socket`)
  is mismatched, so fault accuracy is not a clean headline metric.
- Volume quartiles did not show degradation: largest telemetry quartile still had Hit@1 = 1.0.

RCAEval remains useful for debugging and ablations, but not as the primary "MAS win" proof unless
we pre-register a hard split.

---

## 2. Literature Survey: What Actually Makes MAS Win

### 2.1 Long-context collaboration is the strongest mechanism

**Chain-of-Agents (CoA), NeurIPS 2024** argues that long-context tasks have two failure modes:
retrieval can omit needed evidence, while full-context models can fail to focus. CoA assigns short
segments to worker agents, then uses a manager to integrate evidence, reporting improvements up to
10% over RAG/full-context baselines on long-context tasks.

Takeaway for us: MAS should win only when workers each read evidence that a single context cannot
reliably retain. If a deterministic tool compresses all evidence into one small board, CoA's mechanism
disappears.

Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract-Conference.html

### 2.2 Long contexts are not reliably used by a single model

**Lost in the Middle** shows that models can degrade when relevant information is placed in the
middle of long contexts, even for long-context models. This supports a protocol where the single
baseline receives large raw/semiraw telemetry context and must locate evidence, while MAS workers
each operate on bounded local contexts.

Source: https://arxiv.org/abs/2307.03172

### 2.3 Voting helps only when errors are decorrelated

**Self-Consistency** and **More Agents Is All You Need** show gains from sampling multiple reasoning
paths/agents and voting. But this only helps when samples are meaningfully diverse. If all agents read
the same compact evidence board, errors are highly correlated.

Sources:
- https://arxiv.org/abs/2203.11171
- https://arxiv.org/abs/2402.05120

### 2.4 Debate can hurt

Multi-agent debate can improve reasoning in some settings, but newer failure analyses show that
debate can amplify wrong answers, especially when agents copy/agree with persuasive but incorrect
peers. MAS should avoid free-form debate as the core mechanism; use evidence isolation and targeted
verification instead.

Sources:
- https://arxiv.org/abs/2305.14325
- https://arxiv.org/html/2509.05396v1
- https://arxiv.org/html/2503.13657v1

### 2.5 RCA benchmark implications

**OpenRCA** is the best primary battlefield: it requires analyzing large volumes of metrics, logs,
traces, and dependency structure. The repo explicitly says RCA-Agent uses Python data retrieval to
avoid overly long contexts, and the benchmark expects root-cause time/component/reason.

Source: https://github.com/microsoft/OpenRCA

**RCAEval** is useful but must be split carefully. It has 735 cases across RE1/RE2/RE3; RE2/RE3 have
multi-source telemetry, with large log/trace volumes. RE3's code-level labels need separate handling.

Source: https://arxiv.org/html/2412.17015v1

**TN-AutoRCA** is the strongest telecom-specific direction if data is obtainable. It reports a real
telecom alarm RCA benchmark (TN-RCA530) and an agentic evaluate-analyze-repair loop improving
F1 from a direct LLM baseline to a much higher final score. Even if the dataset is not immediately
available, its design strongly suggests adding cross-case error analysis and systematic rule repair.

Source: https://arxiv.org/html/2507.18190v1

---

## 3. Design v5: Evidence-Isolated ShardRCA + Cross-Case Repair

### 3.1 Name

**ShardRCA-v5: Evidence-Isolated Multi-Agent RCA with Learned Error Repair**

### 3.2 Core idea

The MAS must have a capability a single context does not have:

1. **Evidence isolation:** each worker has exclusive access to a bounded raw telemetry shard.
2. **Local candidate distributions:** each worker emits calibrated candidate likelihoods, not just a
   global top anomaly list.
3. **Cross-shard fusion:** the manager combines independent evidence distributions.
4. **Adversarial verifier:** a separate verifier tries to disprove the selected root using targeted raw
   checks.
5. **Cross-case repair:** after a validation split, an analyzer agent studies bad cases and updates a
   versioned rulebook/scorer before a frozen test run.

The current "single gets the same compact board" design is removed from the headline comparison.

### 3.3 System architecture

```
Task + telemetry catalog
  |
  v
Planner
  - creates disjoint shards by modality/time/component group
  - estimates per-shard evidence density
  - assigns worker budgets
  |
  +--> Metric Worker(s): raw metric windows only
  +--> Log Worker(s): raw log templates / error snippets only
  +--> Trace Worker(s): raw span/latency/error slices only
  +--> Topology Worker: dependency graph / propagation paths only
  |
  v
Evidence Board
  - local candidates with likelihood, negative evidence, evidence pointers
  - no global "top anomaly oracle"
  |
  v
Fusion Manager
  - product-of-experts / Dempster-Shafer style fusion
  - LLM synthesizer only after numeric fusion narrows candidates
  |
  v
Falsifier
  - targeted raw queries against top and runner-up
  - must return SUPPORT / REFUTE / INSUFFICIENT with pointers
  |
  v
Final RCA answer
```

### 3.4 Critical design changes vs v4

| v4 flaw | v5 correction |
|---|---|
| Single baseline receives same global compact board | Single headline baseline gets one context/tool loop and no subagent-generated board |
| Deterministic miner returns global top anomalies | Tools become local and budgeted: query one modality/window/component group at a time |
| MAS workers do not add unique information | Each worker owns a disjoint raw shard and reports both positive and negative evidence |
| RCAEval random split near ceiling | Headline uses OpenRCA Telecom; RCAEval uses a pre-registered hard split |
| Fault taxonomy mismatched for RE3 | Evaluate RE3 root localization separately; map `F1..F5` only with code-level log/trace evidence |
| Debate/vote over same context | Vote only over independent local candidate distributions |

---

## 4. Fair Baselines

We need more than one single baseline, because "single agent" can mean different things.

### 4.1 `single_react`

One LLM context, same primitive tools, same candidate list, same final answer schema.

Rules:

- Cannot call subagents.
- Cannot receive MAS blackboard.
- Can inspect raw shards sequentially through local tools.
- Has a fixed context budget and a fixed tool-call budget.
- If it runs out of budget, it must answer.

This is the main baseline for "single investigator under operational budget."

### 4.2 `single_sc`

Run `single_react` independently k=3 and vote. This controls for self-consistency.

### 4.3 `single_equal_tokens`

A stronger single baseline with the same aggregate token budget as MAS but still one context. This
tests whether MAS wins from independent context memory rather than simply more tokens.

### 4.4 `single_rca_agent_replica`

A code-execution baseline modeled after OpenRCA's RCA-Agent:

- Python/pandas retrieval allowed.
- No subagent parallel memory.
- Same raw data access.
- Same final schema.

This is the hostile-review baseline. If MAS beats only `single_react` but not this, the claim must be
limited to "operational single-agent budget," not "all strong single agents."

---

## 5. Dataset Strategy

### 5.1 Primary: OpenRCA Telecom

Use this as the headline because:

- It is closest to the user's stated telecom RCA goal.
- The task involves metrics/logs/traces/dependency reasoning at real volume.
- Published baselines are weak enough to leave room for improvement.

Required local state:

```
data/openrca/Telecom/query.csv
data/openrca/Telecom/telemetry/
```

Current state: missing. Download/setup is a gating task.

### 5.2 Secondary: RCAEval-Hard

RCAEval random sample is too easy. Define a pre-registered hard split without peeking at model
predictions:

Include cases satisfying at least two of:

- suite is RE2 or RE3;
- system is Train Ticket (`*-TT`);
- telemetry size is in top 50%;
- metric table is wide enough to stress one-pass inspection;
- evidence exists in logs/traces, not only metrics;
- suite is RE3 when evaluating code-level behavior, but report it separately because RE3 uses
  `F1..F5` labels.

Report:

- root Hit@1/Hit@3/MRR for all cases;
- RE1/RE2 fault-family accuracy;
- RE3 code-fault accuracy only after adding explicit `F1..F5` mapping/evidence extraction.

### 5.3 Optional: TN-AutoRCA / TN-RCA530

Check public availability. If obtainable, this may become the best telecom-specific headline because
it is explicitly alarm-based telecom RCA and includes difficulty stratification.

---

## 6. Experiment Protocol

### 6.1 Pre-registration before live run

Write a frozen config file before running:

```
results/prereg_v5_<date>.json
```

It must include:

- dataset split IDs;
- systems and exact budgets;
- model names and temperatures;
- tool schemas and max tool calls;
- primary metric;
- statistical test;
- stopping rule.

No post-hoc dataset filtering after seeing results.

### 6.2 Primary metrics

OpenRCA:

- strict all-or-nothing score from official scoring points;
- component accuracy;
- reason accuracy;
- time accuracy;
- exact paired McNemar vs `single_sc` and `single_rca_agent_replica`.

RCAEval:

- root Hit@1;
- Hit@3;
- MRR;
- fault accuracy only where taxonomy is valid.

### 6.3 Clear-win gate

Declare MAS a clear win only if all hold:

1. `shardrca_v5` beats the strongest single baseline by **>= 10 percentage points** on primary
   accuracy, or by **>= 20% relative error reduction** if baseline accuracy is high.
2. Exact paired McNemar p <= 0.05.
3. `single_sc` does not close the gap.
4. The win persists in the high-telemetry-volume bin.
5. Tokens and latency are reported; if MAS uses more aggregate tokens, the claim is phrased as
   "higher accuracy at higher parallel compute," unless equal-token MAS also wins.

### 6.4 Mechanism tests

Run ablations:

| System | Purpose |
|---|---|
| `no_shard` | tests whether disjoint contexts matter |
| `no_logs` | tests multi-modal evidence |
| `no_traces` | tests trace contribution |
| `no_topology` | tests propagation reasoning |
| `no_falsifier` | tests adversarial verification |
| `no_repair` | tests cross-case repair |
| `same_board_single` | diagnostic oracle: tests whether a single model with the same global board can match or beat MAS |

Expected pattern:

- `same_board_single` matches or beats current MAS if the global board is still the best
  representation; v8 should narrow this gap through local candidate evidence and fusion.
- `no_shard` drops most on long-context/high-volume cases.
- `no_repair` drops on repeated systematic errors.

### 6.5 Executed v7 holdout, 2026-07-03

Pre-registration:

- Config: `results/prereg_v7_holdout20_2026-07-03.json`
- Dataset: `rcaeval_hard`, seed 12, n = 20, no overlap with the exploratory v7 pilot.
- Primary treatment: `rcaeval_shardrca_full`
- Primary baseline: `rcaeval_single_react_sc`
- Primary metric/test: root Hit@1, exact paired McNemar, alpha = 0.05, effect threshold >= +0.10.
- Stopping rule: run exactly the frozen 20 cases once; no p-value extension or post-hoc filtering.

Result:

| System | Hit@1 | Hit@3 | MRR | Fault acc | Total tokens | Avg latency |
|---|---:|---:|---:|---:|---:|---:|
| `rcaeval_shardrca_full` | 0.60 | 0.70 | 0.6813 | 0.15 | 395,048 | 19.736s |
| `rcaeval_single_react` | 0.20 | 0.20 | 0.2000 | 0.05 | 163,718 | 11.534s |
| `rcaeval_single_react_sc` | 0.20 | 0.20 | 0.2000 | 0.05 | 491,085 | 38.743s |
| `rcaeval_same_board_single` | 0.70 | 0.85 | 0.7667 | 0.15 | 365,380 | 30.899s |

Paired primary result against `single_react_sc`:

| Metric | MAS-only correct | single_sc-only correct | Mean diff | 95% CI | exact p |
|---|---:|---:|---:|---:|---:|
| Hit@1 | 8 | 0 | +0.40 | [0.20, 0.60] | 0.007812 |
| Hit@3 | 10 | 0 | +0.50 | [0.30, 0.70] | 0.001953 |
| MRR | 14 | 0 | +0.4813 | [0.2938, 0.6646] | 0.000122 |
| Fault acc | 2 | 0 | +0.10 | [0.00, 0.25] | 0.500000 |

Diagnostic control against `same_board_single`:

- Strongest-single analysis artifact:
  `results/rcaeval_hard_llm_v7_holdout20_analysis_strongest.json`.
- The analyzer now resolves `--baseline strongest_single` to `rcaeval_same_board_single` for this
  holdout.
- `same_board_single` is better by 2 Hit@1 cases: mean difference for MAS is -0.10, exact p = 0.5.
- The current MAS win is therefore against the budgeted single-context investigator, not against a
  single model that receives the same global evidence board/oracle.
- This points directly to v8: workers should emit local candidate evidence and the manager should
  fuse candidate distributions, rather than letting a global board remain the strongest diagnostic
  representation.

Exploratory v8 diagnostic:

- Prototype system: `shardrca_local_fusion`
- Artifact: `results/rcaeval_hard_offline_v8_local_fusion_holdout20.json`
- Mode: offline only, no LLM, not pre-registered evidence.
- Result: Hit@1 = 0.40 / Hit@3 = 0.50 / MRR = 0.5079.
- Interpretation: naive local-candidate fusion is not enough. It improves mechanism separation but
  loses several TrainTicket propagation cases, so the next v8 work should add topology-aware
  propagation constraints and a targeted local verifier before any new live preregistration.

---

## 7. Implementation Plan

Current implementation progress:

- Opaque RCAEval runtime IDs implemented; previous `case_id`/tag label leak is fixed.
- Catalog path leakage removed from model-facing telemetry summaries; v6 live runs are invalid and
  v7 is the first label-safe live namespace after this fix.
- `rcaeval_hard` metadata split implemented with 368 label-safe hard cases.
- `single_react` / `single_react_sc` implemented as budgeted local-tool baselines.
- `same_board_single` implemented as a diagnostic control for the old v4 tie.
- `result_analysis.py` implemented for paired deltas, disagreement counts, and usage summaries.
- v7 holdout result saved and analyzed:
  `results/rcaeval_hard_llm_v7_holdout20.json`,
  `results/rcaeval_hard_llm_v7_holdout20_analysis_sc.json`.
- Exploratory `shardrca_local_fusion` ablation implemented; offline diagnostic is negative and
  should not be used as headline evidence.

### Phase 0 -- Stop running weak comparisons

Do not spend more API budget on current `shardrca_full,single,single_sc` over random RCAEval. It
has already proven the old design ties.

### Phase 1 -- Add hard split and analysis utilities

Files:

- `telco_mas/shardrca/hard_split.py`
- `telco_mas/shardrca/result_analysis.py`
- `tests/test_shardrca_hard_split.py`

Deliverables:

- deterministic `rcaeval_hard` case list;
- telemetry volume bins;
- simple-profile rank metadata;
- result analyzer that prints paired deltas and disagreement examples.

Gate:

- hard split has at least 100 cases, or if fewer, all qualifying cases are reported.
- Current gate status: **passed** (`results/rcaeval_hard_split.json` contains 368 cases).

### Phase 2 -- Replace global mining with budgeted local tools

Files:

- `telco_mas/shardrca/local_tools.py`
- update `single_baseline.py`
- update `miner.py`

Tool constraints:

- `query_metric_shard(modality/window/components, limit)`
- `query_log_shard(window/components/pattern, limit)`
- `query_trace_shard(window/components, limit)`
- no tool may return a global all-component top anomaly board.

Gate:

- `same_board_single` reproduces v4 tie;
- new `single_react` no longer receives the MAS board.
- Current gate status: **passed for the operational single-context comparison**. The v7 holdout shows
  `shardrca_full` clearly beats `single_react` and `single_react_sc`; however, the diagnostic
  `same_board_single` remains stronger than MAS, so the architecture still needs local-candidate
  fusion before claiming a mechanism-level MAS advantage over a global-board single oracle.

### Phase 3 -- Evidence-isolated MAS

Files:

- `telco_mas/shardrca/fusion.py`
- `telco_mas/shardrca/evidence.py`
- update `board.py`, `planner.py`, `runner.py`

Worker output schema:

```
CandidateEvidence {
  component,
  reason_family,
  support_score,
  refute_score,
  modality,
  shard_id,
  evidence_ptrs,
  missing_evidence,
  local_rank
}
```

Fusion:

- combine independent likelihoods;
- penalize unsupported single-modality candidates;
- boost convergent multi-modal evidence;
- preserve uncertainty.

### Phase 4 -- Cross-case repair

Files:

- `telco_mas/shardrca/repair_loop.py`
- `telco_mas/shardrca/rulebook.py`

Protocol:

- train/validation/test split;
- run frozen v5 on validation;
- bad-case analyzer clusters errors;
- LLM coder proposes rulebook/scorer changes;
- reviewer checks label leakage;
- freeze rulebook;
- run once on test.

This follows TN-AutoRCA's key insight: systematic feedback over bad cases can improve domain RCA
more than isolated per-case prompting.

### Phase 5 -- OpenRCA Telecom

Tasks:

- download/extract OpenRCA Telecom;
- cache window extracts;
- implement official prompt fields/candidate constraints;
- run `single_react`, `single_sc`, `single_rca_agent_replica`, `shardrca_v5`.

Gate:

- n >= 40, preferably all Telecom tasks.

---

## 8. Risk Register

| Risk | Mitigation |
|---|---|
| MAS still ties because tools over-compress evidence | Enforce local tool outputs; add `same_board_single` diagnostic |
| Single equal-token catches up | Report honest boundary; keep operational equal-latency claim separate |
| OpenRCA data unavailable | Use RCAEval-Hard as secondary and pursue TN-AutoRCA availability |
| RE3 fault labels remain mismatched | Separate RE3 root metric from fault-family metric; implement `F1..F5` evidence extraction |
| Reviewer calls hard split cherry-picking | Define hard split by pre-results telemetry properties only; include full-suite appendix |
| MAS wins only by spending more tokens | Report tokens/latency and include equal-token/equal-latency rows |

---

## 9. Immediate Next Commands

1. Reproduce the registered v7 analysis:

```bash
python3 -m telco_mas.shardrca.result_analysis \
  results/rcaeval_hard_llm_v7_holdout20.json \
  --baseline strongest_single \
  --treatment rcaeval_shardrca_full
```

2. Inspect the remaining oracle gap:

```bash
python3 -m telco_mas.shardrca.result_analysis \
  results/rcaeval_hard_llm_v7_holdout20.json \
  --baseline rcaeval_same_board_single \
  --treatment rcaeval_shardrca_full
```

3. Implement v8 local-candidate fusion:

- each worker returns `CandidateEvidence` records from its shard;
- the manager fuses candidate distributions instead of consuming a global blackboard;
- `same_board_single` remains a diagnostic oracle, not the headline baseline.
- current prototype `shardrca_local_fusion` is implemented but underperforms, so add topology-aware
  propagation and local verification before spending on live LLM runs.

4. Before spending on another live holdout, freeze a new preregistration because v8 changes the
algorithm:

```bash
python3 -m telco_mas.shardrca.hard_split --out results/rcaeval_hard_split.json
```

5. Run regression tests after implementation changes:

```bash
pytest -q tests/test_shardrca.py tests/test_rcaeval_adapter.py
python3 -m compileall -q telco_mas
```

6. In parallel, download OpenRCA Telecom and run the same protocol there.

---

## 10. Success Definition

The goal is not complete until we have:

- a system where MAS and single are not accidentally sharing the same evidence board;
- a pre-registered hard or telecom split;
- a live result where MAS beats the strongest single baseline by the clear-win gate;
- ablations showing the win comes from evidence isolation/fusion/verification, not arbitrary
  prompt variance;
- report tables and plots with token/cost accounting.

Current status against this definition: the first three are substantially satisfied for the
budgeted single-context baseline on RCAEval-Hard v7. The goal is **not fully complete** until
mechanism ablations are run, OpenRCA/TN-style external validity is addressed, and the report tables
make the same-board oracle limitation explicit.

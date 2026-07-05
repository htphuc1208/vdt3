# PLAN v6 -- Real-Benchmark-First MAS RCA Protocol

**Objective:** make a MAS beat strong single-agent RCA baselines clearly, preferably on real telecom
or real operational RCA benchmarks. Synthetic data is allowed only as a fallback, and only if the
dataset design matches telecom RCA practice and is justified by published benchmark methodology.

**Update 2026-07-05 -- fresh-holdout replication PASSED.** The "MAS beats a budgeted single
agent" result now replicates on a fresh, fully disjoint RCAEval-hard holdout (n=24, disjoint from
the v7 holdout, the seed99 repro, and the high-volume pilot). Frozen preregistration:
`results/prereg_rcaeval_fresh_confirm24_frozen.json`; result:
`results/rcaeval_hard_llm_fresh_confirm24.json`. ShardRCA Hit@1 = 0.667 vs `single_react_sc` 0.375
(+0.292, exact paired McNemar p = 0.039, 8 MAS-only wins vs 1 baseline-only win); identical gate
vs `single_react`. The global-board oracle `same_board_single` scores 0.708 and ties MAS (p = 1.0,
0 MAS-only wins vs 1 oracle-only win) -- this is a documented boundary, not the confirmatory gate.
Two mechanisms landed this turn: (1) the topology/temporal causal re-rank is now wired into the
live MAS synthesize path (previously only on the OpenRCA path), loaded from frozen weights
`results/weights/local_fusion_fit_v3_temporal.json`; (2) three latent telecomts domain-guard
signatures were wired in (all 208 tests pass). The honest boundary stands: on small microservice
cases the whole evidence board fits one context, so evidence-isolation gives no edge over an oracle
single agent. A real-telecom *headline* still needs TN-RCA530 (no official artifact found on
2026-07-05 despite a live web search) or authorized gated telco datasets; OpenRCA Telecom remains
directionally positive (MAS ranks first over the SOTA-style RCA-Agent replica) but underpowered.

**Current local state, 2026-07-04:** OpenRCA Telecom is now present locally and the prepared
strict-data path has run. The frozen paired run
(`results/openrca_paired_frozen.json`, analyzed in
`results/openrca_paired_frozen_analysis.json`) failed the confirmatory gate: ShardRCA reached
strict 0.10 versus 0.04 for `single_react_sc`, 0.04 for `same_board_single`, and 0.04 for
`rca_agent_replica`, but the Holm-adjusted confirmatory p-value is 0.5 and the high-volume bin is
0% for all systems. This run is now consumed diagnostic evidence, not reusable confirmatory
evidence after algorithm changes.

RCAEval-Hard v7 remains useful evidence that ShardRCA can beat a budgeted single-context baseline,
but it is not telecom-specific and must not become the final headline proof. TelecomTS test evidence
also failed its synthetic fallback gate and is consumed diagnostic evidence.

The latest telco-v3 development run,
`results/synthetic_telco_v3_full_vs_single_power_expert.json`, is also negative evidence:
full MAS solved 8/12 strict diagnoses (0.667), while the strong single solved 9/12 (0.750).
The paired delta is -0.0833 (95% paired bootstrap CI [-0.3333, 0.1667], exact McNemar
`p=1.0`), and MAS used 2.41x the tokens. Adding the power/site expert coincided with a
full-MAS improvement over an earlier run, but it did not establish causality or an advantage
over the simultaneously rerun single baseline.

The later verifier-development run, `results/synthetic_telco_v3_verifier_dev.json`, improved
strict diagnosis to 11/12 for full MAS versus 10/12 for single, but still failed the clear-win gate:
paired strict delta `+0.0833`, exact McNemar `p=1.0`; end-to-end delta `0.0`; full used 2.17x the
single baseline tokens. `results/synthetic_telco_v3_verifier_ablation_analysis.json` shows no
strict/end-to-end difference between `full` and `no_verifier`, so verifier contribution is not
established.

The detailed source survey and benchmark ladder are in `BENCHMARK_SURVEY_REAL_TELCO.md`.

Post-review contamination rule: the reviewed OpenRCA 50/51 run, TelecomTS 39-event run, and
RCAEval-Hard v7 n=20 holdout may be used for error analysis, floor calibration, documentation, and
algorithm debugging only. Any repaired algorithm requires a new frozen validation/confirmatory
split or a new external benchmark before supporting a full claim.

---

## 1. Benchmark Survey And Priority

### Tier A -- TN-RCA530 / TN-AutoRCA

Best scientific fit for the goal if obtainable.

- Domain: real telecommunication network alarm RCA.
- Scale: 530 fault scenarios.
- Representation: knowledge graphs over physical equipment/topology plus alarm data.
- Task: identify root cause node(s), equipment ID, and proposed solution from graph-structured alarm
  evidence.
- Primary metric in the paper: macro F1 over predicted root-cause tuples.
- Why it matters: the paper explicitly argues telecom RCA is graph-structured, multi-label, and
  difficult for direct LLMs; this is exactly where MAS/tool/agent architecture should have a real
  mechanism.
- Current local status: `source_only_no_artifact`. The 2026-07-04 source refresh found the
  arXiv/OpenReview/Hugging Face paper pages, but no official TN-RCA530 dataset URL, local files, or
  schema are configured in this workspace.

Action:

- Obtain an official dataset release URL, author-provided files, or a documented contact route. The
  arXiv paper states the benchmark is public, but the inspected source pages did not expose a
  runnable artifact.
- If obtained, implement a `telco_mas.tnrca` adapter before any algorithm tuning.
- Do not reconstruct TN-RCA530 from prose or synthetic lookalikes; that would be survey support, not
  headline evidence.
- Do not use RCAEval holdout failures to tune a TN-RCA method.

Sources: https://arxiv.org/html/2507.18190v1, https://arxiv.org/abs/2507.18190,
https://openreview.net/forum?id=s5mwg63B02, https://huggingface.co/papers/2507.18190

### Tier B -- OpenRCA Telecom

Best currently integrated real-operational benchmark path.

- Domain: software/operations RCA, with a Telecom split in the expected local layout.
- Data: natural-language queries plus telemetry: KPI time series, dependency traces, and logs.
- Scale/resource: upstream README recommends about 80GB storage and 32GB memory.
- Why it matters: it requires multi-modal telemetry retrieval and dependency reasoning. It is not
  alarm-only telecom RCA, but it is much closer to real operations than synthetic toy suites.

Current repo support:

- Loader/evaluator exist under `telco_mas/openrca/`.
- Preregistration generator exists under `telco_mas/openrca/prereg.py`; it freezes `query.csv`
  SHA-256, a telemetry manifest hash, exact row IDs, difficulty bins, systems, metrics, tests, and
  commands using `--row-ids`.
- Cross-benchmark readiness checker exists under `telco_mas/evaluation/benchmark_readiness.py`; it
  verifies local data and confirms preregistration hashes still match current files before live runs.
- Expected local layout:

```text
data/openrca/Telecom/query.csv
data/openrca/Telecom/telemetry/
```

Action:

- Treat the existing 50/51-row frozen run as consumed diagnostic evidence.
- Generate `results/openrca_error_taxonomy.json` and run the no-LLM `heuristic_floor` baseline to
  identify whether the failure is parsing/format, telemetry mining, or reasoning.
- Fit any repair weights only on declared validation rows that do not overlap a frozen
  confirmatory preregistration.
- Freeze a new preregistered run only after the algorithm, weights, row IDs, candidate-catalog
  provenance, and ablations (`no_falsifier`, `no_topology`, `no_refinement`) are fixed.

Source: https://github.com/microsoft/OpenRCA

### Tier B2 -- OpenRCA 2.0

High-priority future operational RCA benchmark if official artifacts become available.

- Domain: cross-system RCA.
- Scale: 500 instances.
- Representation: step-wise causal propagation annotations produced by PAVE.
- Metrics: path reachability, node F1, edge F1, plus exact root-cause set recovery.
- Why it matters: this directly tests whether agents can ground a diagnosis in a verified causal
  path, which is a stronger scientific target than root-name matching.

Action:

- Track official artifact release.
- If data is available, implement a separate adapter rather than squeezing it into OpenRCA 1.0
  scoring.
- Pre-register causal-path metrics before any run.

Source: https://arxiv.org/abs/2606.27154

### Tier C -- TeleLogs / TeleLogsAgent

Best synthetic 5G fallback, not a replacement for real telecom alarm RCA.

- Domain: 5G wireless troubleshooting.
- Synthetic but telecom-specific: drive-test scenarios with UE mobility, gNodeB/cell configuration,
  RSRP, SINR, throughput, neighbor cells, antenna azimuth/downtilt, and resource allocation.
- Task: diagnose throughput degradation and root causes among predefined causes.
- TeleLogsAgent adds tool-use evaluation over 5G scenarios (`TS1`, `TS2`, `TS3`) and is therefore a
  closer fallback for agentic MAS experiments than text-only classification.
- Why it matters: if real TN-RCA/OpenRCA cannot be obtained quickly, TeleLogs is the most defensible
  fallback because its variables resemble network-engineering diagnosis rather than generic software
  incidents.

Current repo support:

- `telco_mas/telelogs_agent/dataset.py` validates local gated data laid out as:

```text
data/telelogs_agent/TS1/test.json
data/telelogs_agent/TS2/test.json
data/telelogs_agent/TS3/test.json
```

- `telco_mas/telelogs_agent/prereg.py` freezes split counts, per-file SHA-256 hashes, manifest hash,
  selected row IDs per TS split, systems, metrics, and stopping rule.
- `telco_mas/evaluation/benchmark_readiness.py` checks whether fallback data and preregistration
  hashes still match before any fallback run.
- `scripts/download_telelogs_agent.sh` downloads the gated Hugging Face files after the dataset
  terms are accepted and `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is available.
- `scripts/download_netop_telco_dataset.sh` can also fetch gated `telelogs`, `telelogs_agent`, or
  `telco_troubleshooting_challenge` repositories after terms are accepted.
- `telco_mas/telelogs/` implements the official TeleLogs text RCA fallback path: flexible JSON/JSONL
  loader, manifest/row preregistration, label-safe `profile`/`llm` runner, exact structured
  root-cause-set scoring, paired analysis, and readiness integration.
- `telco_mas/telelogs_agent/cli.py` can produce staged `profile` or label-safe `llm` result files
  over preregistered rows, and `--mode tool` adapts the official TeleLogsAgent FastAPI server into
  OpenAI-style tool calls with `X-Scenario-Id`. The tool runner records task success, average tool
  calls, tool-call failure rate, and score-per-tool-call efficiency.
- `telco_mas/telelogs_agent/result_analysis.py` reports paired effects and the same clear-win gate
  shape used elsewhere.
- Runtime task loading strips label-like keys such as `answer`, `expected`, `ground_truth`,
  `root_cause`, `solution`, and `label`.

Action:

- Accept dataset terms at https://huggingface.co/datasets/netop/TeleLogsAgent and export an HF token,
  then run `scripts/download_telelogs_agent.sh`.
- Treat as fallback/secondary evidence.
- Preserve official train/test split.
- Do not train or tune on official test.
- Report it as synthetic 5G drive-test evidence, not real telecom operations evidence.
- Run readiness after freezing preregistration and before any fallback experiment.
- Treat staged `profile`/`llm` runner output as fallback engineering evidence only. Prefer
  `--mode tool` because it exercises the official TeleLogsAgent HTTP tools; it still requires the
  gated code/data and a running FastAPI server.
- Treat TeleLogs as official synthetic 5G fallback evidence only after the gated files are present,
  a `test` split preregistration matches current hashes, and the paired analyzer is run. Treat Telco
  Troubleshooting Agentic Challenge as a tracked candidate only until a separate label-safe
  adapter/evaluator is implemented.

Sources:

- https://huggingface.co/datasets/netop/TeleLogs
- https://huggingface.co/datasets/netop/TeleLogsAgent
- https://arxiv.org/pdf/2507.21974

Local access check: on 2026-07-03, unauthenticated download failed with Hugging Face gated-repo
`401`; no TeleLogsAgent evidence is available locally yet.

Staged run commands after access is available:

```bash
scripts/download_telelogs_agent.sh
python3 -m telco_mas.telelogs_agent.prereg \
  --limit-per-set 20 \
  --systems single_react_sc,shardrca_full \
  --out results/prereg_telelogs_agent_frozen.json

# In another shell, from data/telelogs_agent. For all TS1/TS2/TS3 in one run, start one
# FastAPI server per TELELOGS_AGENT_CONFIG/port and replace --server-url with:
# --server-url-map TS1=http://localhost:7861,TS2=http://localhost:7862,TS3=http://localhost:7863
# TELELOGS_AGENT_CONFIG=TS1 python fastapi_server.py
python3 -m telco_mas.telelogs_agent.cli \
  --prereg results/prereg_telelogs_agent_frozen.json \
  --mode tool \
  --server-url http://localhost:7861 \
  --out results/telelogs_agent_tool.json
python3 -m telco_mas.telelogs_agent.result_analysis \
  results/telelogs_agent_tool.json \
  --baseline strongest_single \
  --treatment shardrca_full \
  --out results/telelogs_agent_tool_analysis.json

# Lower-evidence staged fallback if the FastAPI server is unavailable.
python3 -m telco_mas.telelogs_agent.cli \
  --prereg results/prereg_telelogs_agent_frozen.json \
  --mode llm \
  --out results/telelogs_agent_staged_llm.json
python3 -m telco_mas.telelogs_agent.result_analysis \
  results/telelogs_agent_staged_llm.json \
  --baseline strongest_single \
  --treatment shardrca_full \
  --out results/telelogs_agent_staged_llm_analysis.json
```

### Tier D -- RCAEval-Hard

Useful supporting benchmark, not final telecom proof.

- Domain: microservice RCA across Online Boutique, Sock Shop, and Train Ticket.
- Data: 735 failure cases; RE2/RE3 include logs/traces/metrics.
- Current evidence: v7 preregistered RCAEval-Hard holdout shows MAS beats budgeted single-context
  `single_react_sc` on root Hit@1 by +0.40 with exact McNemar p = 0.007812.
- Limitation: it is not telecom; also `same_board_single` remains stronger than MAS on the holdout,
  so it is a mechanism diagnostic, not the final claim.

Action:

- Keep RCAEval only for engineering smoke, label-safety tests, and non-headline ablations.
- Freeze the v7 holdout; do not use it for further tuning.

Sources:

- https://github.com/phamquiluan/RCAEval
- https://zenodo.org/records/14590730

---

## 2. Scientific Guardrails

These are binding for future evidence:

1. **No post-hoc holdout tuning.** Once a holdout has been used for a result, it becomes locked.
   Any algorithm changes after seeing that result require a new validation/holdout split and a new
   preregistration.
2. **Real benchmark first.** The headline must be TN-RCA530 or OpenRCA Telecom if data is obtainable.
   RCAEval can be supporting evidence only.
3. **Synthetic fallback must be benchmark-like.** It must follow telecom RCA structure: topology,
   alarm floods, KPI windows, causal propagation, expert-verifiable labels, and difficulty bins.
4. **Same access, different organization.** MAS and single baselines must use the same primitive data
   access. MAS may partition evidence across workers; single may inspect sequentially under its
   budget. A same-board single remains a diagnostic oracle, not a fair operational single baseline.
5. **All label-bearing fields are evaluator-only.** Runtime prompts/tools may include opaque case IDs,
   telemetry summaries, topology, alarms, logs, traces, and candidate universes, but not label-derived
   path names, root causes, fault names, or scoring points.
6. **Report compute honestly.** If MAS uses more aggregate tokens/tools than a single agent, the claim
   is "higher accuracy at higher parallel compute" unless an equal-token/equal-wall-clock comparison
   also wins.
7. **Claim audit is mandatory.** Before writing any MAS-win claim, run
   `python3 -m telco_mas.evaluation.claim_audit --strict` on the current readiness report and paired
   analysis artifact. A failed audit means the result is calibration/supporting evidence only.

---

## 3. Baselines Required Before A Clear Win Claim

Minimum baselines:

- `single_react`: one context, same primitive tools, fixed tool budget, no subagents.
- `single_react_sc`: k independent single runs with vote/self-consistency.
- `single_equal_tokens`: one-context baseline with the same aggregate token budget as MAS where the
  benchmark/tooling supports it.
- `code_retrieval_single`: one single-agent Python/pandas retrieval baseline, modeled after OpenRCA's
  RCA-agent idea that code-based retrieval avoids overly long contexts.
- `same_board_single`: diagnostic oracle. If this beats MAS, the mechanism claim must say that the
  global compact board remains stronger than current MAS adjudication.

MAS must beat at least the strongest operational single baseline by the preregistered primary metric.
The analyzer resolves `--baseline strongest_single` deterministically: among frozen systems with
single-agent names and paired cases against the treatment, choose highest strict accuracy, then
highest partial score, then lower token usage, then lexical system name. This prevents cherry-picking
`single_react_sc` when another registered single baseline is stronger.
Beating only a weak direct-prompt baseline is not enough.

Current implementation status: `single_react`, `single_react_sc`, `single_equal_tokens`,
`code_retrieval_single`, and `same_board_single` are registered in the ShardRCA/OpenRCA runners. The
TelecomTS runner also keeps a development-only shape-prototype diagnostic disabled by default; it
must not be promoted into the claim protocol unless validation shows source-session generalization.
The current TelecomTS MAS domain guard is above the strongest single baseline on a nine-event
validation-source calibration slice, but the exact paired gate remains non-significant. Those events
were inspected during iterative guard tuning, so they are development evidence rather than an
unbiased held-out result; the algorithm remains unlocked and test stays untouched.

Calibration contamination ledger (2026-07-04): validation event indices `0` through `8` and their
labels were inspected while developing domain-guard versions through v4. They are permanently
consumed for calibration, must not support a confirmatory claim, and must not drive further guard
tuning. The observed macro delta `+0.4285` and exact McNemar `p=0.125` are diagnostic only. Because
interference and temporal RF-filter cases remain unresolved, the preferred next step is fresh
source-session validation from another dataset; running TelecomTS test is deferred until an
algorithm is frozen and preregistered.
expanded-budget single is a compute-sensitivity diagnostic; `code_retrieval_single` is the hostile
single-agent code-retrieval baseline.

---

## 4. Experimental Protocol

### 4.1 OpenRCA Telecom

Pre-registration artifact:

```text
results/prereg_openrca_telecom_<date>.json
```

Must include:

- exact dataset root and validation checksum/row count;
- row IDs, or all rows if feasible;
- system list and budgets;
- model, temperature, cache state;
- primary metric: strict OpenRCA score or strict accuracy, plus partial score;
- paired tests: exact paired sign/McNemar where applicable, bootstrap CI for score deltas;
- stopping rule: fixed rows, no extension by p-value.

Run ladder:

```bash
python3 -m telco_mas.openrca.cli --strict-data --limit 1
python3 -m telco_mas.openrca.prereg \
  --limit 40 \
  --contaminated-row-ids all \
  --out results/prereg_openrca_telecom_frozen.json
python3 -m telco_mas.evaluation.benchmark_readiness \
  --strict \
  --out results/benchmark_readiness.json
python3 -m telco_mas.openrca.cli \
  --mode llm \
  --confirm-live-llm \
  --systems single_react_sc,shardrca_full \
  --limit 3 \
  --out results/openrca_smoke_paired.json
```

Then run the exact frozen paired experiment:

```bash
python3 -m telco_mas.openrca.cli \
  --mode llm \
  --confirm-live-llm \
  --prereg results/prereg_openrca_telecom_frozen.json \
  --out results/openrca_paired_frozen.json
```

Then run the paired analyzer:

```bash
python3 -m telco_mas.openrca.result_analysis \
  results/openrca_paired_frozen.json \
  --baseline strongest_single \
  --treatment shardrca_full \
  --out results/openrca_paired_frozen_analysis.json
python3 -m telco_mas.evaluation.claim_audit \
  --readiness results/benchmark_readiness.json \
  --analysis results/openrca_paired_frozen_analysis.json \
  --strict
```

The frozen run should include at least 40 rows, preferably the full Telecom split if cost and runtime
allow.

### 4.2 TN-RCA530

Adapter requirements:

- only start after official TN-RCA530 files or schema are available;
- parse KG nodes/edges, alarm nodes, candidate root causes, solutions;
- expose only graph/alarm input at runtime;
- keep true root tuples evaluator-only;
- compute macro Precision/Recall/F1 exactly like the paper's root-cause tuple matching.

Method expectation:

- MAS workers specialize by graph neighborhood/alarm family/equipment layer.
- Fusion manager combines candidate root-cause tuples and checks graph path consistency.
- Repair loop may be used only on training/validation, never on the final test.

### 4.3 TelecomTS Public Testbed-Backed Synthetic Track

TelecomTS is now downloaded and adapted as an intermediate public benchmark. It contains KPI
measurements from a controlled 5G testbed, but its published RCA task excludes the real jamming
sessions and classifies ten synthetic anomaly types. It is therefore below TN-RCA530/OpenRCA and
must never be described as real-fault RCA.

Protocol:

- development sessions: `Zone_A/File`, `Zone_B/Twitch`, `Zone_C/YouTube`;
- validation sessions: `Zone_A/Twitch`, `Zone_B/YouTube`, `Zone_C/File`;
- untouched test sessions: `Zone_A/YouTube`, `Zone_B/File`, `Zone_C/Twitch`;
- split unit is the complete source session, not a randomly sampled 128-point window;
- primary metric is macro root-cause accuracy over the ten official synthetic RCA classes;
- secondary metrics are micro/per-class accuracy, paired exact McNemar, tokens, calls, and latency;
- the upstream 363 held-out stride-32 windows collapse to 39 independent anomaly events after
  interval clustering and overlap deduplication; the test evaluates all 39 events;
- two rare classes have only one test event each, so class-wise estimates are explicitly unstable;
- freeze requires a final algorithm ID and fixed strongest-single candidate set; no test execution
  is allowed while `results/prereg_telecomts_draft.json` remains `draft`.

Runtime label safety excludes the anomaly object/type, affected KPI list, troubleshooting ticket,
Q&A answers/reasoning, generated description/statistics, source path, absolute timestamps, and
`anomaly_present`. The fixed candidate class universe is allowed because the upstream task is
closed-set classification.

Current commands:

```bash
scripts/download_telecomts.sh
python3 -m telco_mas.telecomts.prereg \
  --out results/prereg_telecomts_draft.json
python3 -m telco_mas.telecomts.cli \
  --split development --mode profile --limit 1 \
  --out results/telecomts_development_profile.json
python3 -m telco_mas.telecomts.result_analysis \
  results/telecomts_development_profile.json \
  --baseline strongest_single --treatment telecomts_shardrca_full \
  --out results/telecomts_development_profile_analysis.json
python3 -m telco_mas.evaluation.benchmark_readiness \
  --out results/benchmark_readiness.json
```

Current readiness is deliberately false while the test preregistration is draft. Development and
validation runs may refine the generic method; test cannot run until the algorithm ID, model,
temperature, systems, event IDs, manifest, and no-cache policy are frozen. A future TelecomTS win
can pass claim audit only with `--allow-synthetic` and supports a synthetic-only statement.

### 4.4 Synthetic Telecom Fallback

If real benchmark data remains unavailable, build a synthetic benchmark only after freezing the
schema below.

Required telecom structure:

- network topology: RRU/AAU, BBU/DU/CU, transport/backhaul, core, OSS/NMS;
- alarm stream: timestamps, severity, equipment ID, alarm name, vendor/domain, clear time;
- KPI windows: throughput, RSRP, SINR, PRB utilization, handover failure, drop rate, attach failure,
  CPRI/eCPRI errors, packet loss, CPU/mem/disk for network functions;
- causal graph: root cause -> intermediate cause(s) -> alarm/KPI symptoms;
- long-tail roots: common causes and rare causes;
- difficulty bins: simple one-hop, difficult multi-hop, mixed/noisy, missing-observation cases.

Construction principles, borrowed from TN-RCA530:

- **Veracity proxy:** use real telecom variable names, topology layers, and alarm semantics from
  public standards/docs where possible.
- **Comprehensiveness:** cover common and long-tail root families, not just the cases the MAS likes.
- **Verifiability:** labels come from simulator causal graph, with invariant checks.
- **Complexity discriminability:** compute objective difficulty bins from path ambiguity, alarm fanout,
  distractor count, missing telemetry, and multi-root cases.

Synthetic success is not enough for the final claim unless real data cannot be obtained and the
report clearly says the result is a telecom-valid simulation.

Current development exporter:

```bash
python3 -m telco_mas.synthetic_telco.dataset \
  --suite telco_v3 \
  --out results/synthetic_telco_v3_dataset.json
python3 -m telco_mas.synthetic_telco.prereg \
  --dataset results/synthetic_telco_v3_dataset.json \
  --out results/prereg_synthetic_telco_v3_frozen.json
python3 -m telco_mas.evaluation.benchmark_readiness \
  --out results/benchmark_readiness.json
```

The exported artifact contains label-safe runtime payloads plus evaluator-only labels/causal graphs.
Readiness requires the preregistration hash and runtime case IDs to match the exported artifact.
It is a **last-resort synthetic fallback**, ranked below OpenRCA, TN-RCA530, and TeleLogsAgent.

Post-oracle-fix calibration:

```bash
python3 -m telco_mas.evaluation.telco_result_analysis \
  results/synthetic_telco_v3_full_vs_single_kpi_policy.json \
  --out results/synthetic_telco_v3_full_vs_single_kpi_policy_analysis.json
```

That first calibration failed the clear-win gate (`full - single = -0.1667` strict diagnosis; exact
paired `p=0.6875`). The later power-expert run also failed (`full - single = -0.0833`, exact
`p=1.0`). Telco-v3 has now been inspected during development and must never be presented as an
unseen confirmatory holdout.

### 4.4.1 Telco-v4 Confirmatory Synthetic Holdout

Telco-v4 is implemented but **has not been generated or evaluated as a frozen artifact yet**. It is
valid only after the MAS algorithm is frozen.

Design:

- 56 scenarios: 14 supported fault families x 4 nuisance profiles.
- Every family receives exactly one complete view, alarm-distractor view, incomplete alarm view, and
  knowledge-base holdout view. This crossed allocation prevents fault family from being confounded
  with nuisance difficulty.
- Root placement is shuffled only among topology-valid elements using one declared seed.
- Case IDs are opaque and descriptions contain symptoms rather than evaluator labels.
- Runtime telemetry includes deterministic event times. Root evidence precedes propagated site,
  downstream, and all-RAN effects.
- Runtime and evaluator labels are separated recursively. The artifact commits to all cases with a
  canonical content SHA-256; the runner also verifies the file SHA-256 and reconstructs the runtime
  against current simulator code.
- Multi-root incidents are excluded from the primary confirmatory estimand because the current
  system emits one root. Multi-root set-F1 requires a separately designed output contract.

Scientific basis:

- [3GPP TS 28.111 / ETSI TS 128 111](https://www.etsi.org/deliver/etsi_ts/128100_128199/128111/19.03.00_60/ts_128111v190300p.pdf)
  supplies the fault/error/failure distinction and alarm identity, severity, probable-cause, and
  lifecycle concepts.
- [3GPP TS 28.554 / ETSI TS 128 554](https://www.etsi.org/deliver/etsi_ts/128500_128599/128554/18.09.00_60/ts_128554v180900p.pdf)
  grounds the use of accessibility, integrity, utilization, latency, throughput, and reliability
  measurements.
- [TN-AutoRCA / TN-RCA530](https://arxiv.org/abs/2507.18190) motivates topology/knowledge-graph
  propagation, long-tail root families, verifiable labels, and objective complexity.
- [TeleLogs](https://arxiv.org/abs/2507.21974) supports telecom-specific randomized variants and
  structured diagnostic explanations, while remaining synthetic rather than real-operator proof.
- [NIST IR 8213](https://csrc.nist.gov/pubs/ir/8213/ipd) defines the external randomness-beacon
  format used to make seed selection auditable.

Freeze/run order:

1. Tune and test algorithm changes only on telco-v3 and non-v4 development data.
2. Commit the final algorithm and record the clean commit SHA as `ALGORITHM_ID`.
3. Before its publication, record one future NIST beacon pulse URL/time. After publication, derive
   one integer `SEED` from that pulse. Do not inspect multiple seeds.
4. Generate one v4 artifact with both `--seed` and `--seed-source`.
5. Preregister artifact SHA, content SHA, case order, systems, model, three runs, cache disabled,
   algorithm ID, metrics, and stopping rule.
6. Run the exact command emitted by preregistration with `TELCO_TEMPERATURE=0`. The runner rejects
   any dataset, case order, system order, algorithm ID, model, endpoint, temperature, tool-iteration,
   run-count, or cache-policy mismatch.

```bash
python3 -m telco_mas.synthetic_telco.dataset \
  --suite telco_v4 \
  --seed "$SEED" \
  --seed-source "$BEACON_PULSE_URL" \
  --out results/synthetic_telco_v4_holdout.json

python3 -m telco_mas.synthetic_telco.prereg \
  --dataset results/synthetic_telco_v4_holdout.json \
  --systems full,single,no_verifier,no_repair \
  --runs 3 \
  --model "$OPENAI_MODEL" \
  --temperature 0 \
  --algorithm-id "$ALGORITHM_ID" \
  --out results/prereg_synthetic_telco_v4_frozen.json
```

Primary estimand: scenario-level strict diagnosis accuracy difference, full MAS minus strong single.
Repeated LLM runs are averaged within scenario for the paired effect and reduced by within-scenario
majority for exact McNemar; they are not counted as 168 independent cases. The predeclared clear-win
gate remains absolute delta >= 0.10 **and** exact paired `p<=0.05`, with full token/call/latency
accounting. Fault-family and nuisance-profile results are descriptive diagnostics, not additional
post-hoc claims.

Current development mechanism (not yet a benchmark result):

- `telco_mas/knowledge/fault_ontology.py` separates canonical root-fault families from alarm
  condition names. The same output normalization is applied to MAS and the strong single baseline.
- The MAS evidence verifier rejects a canonical fault assigned to an impossible equipment domain
  and may add a candidate only when a high-specificity root condition occurs at the earliest event
  time. Generic propagated conditions such as `HIGH_INTERFERENCE` and `DEGRADED_QOS` cannot seed a
  candidate.
- `no_verifier` removes candidate seeding and structural veto while retaining shared label
  normalization. This is the required mechanism ablation.
- A bounded repair loop replans once when validation shows the selected root is still faulty;
  `no_repair` isolates its contribution. Independent secondary faults do not trigger a retry.
- The incomplete-alarm v4 profile removes direct root alarms. If a fault has no propagated alarm,
  it exposes a generic service-impact symptom on a related element instead of retaining the root
  alarm.
- RAN/core expert scopes now explicitly cover GNSS/PTP timing loss and UPF user-plane degradation.
  These changes are fault-family level and contain no scenario IDs.

These edits were motivated by v3 development failures, so any improved v3 score remains calibration
evidence only. Their confirmatory value must come from a later frozen real benchmark or untouched v4
artifact.

---

## 5. Immediate Next Steps

1. Continue acquiring OpenRCA Telecom or TN-RCA530. Current local OpenRCA check is blocked by
   missing data; current TN-RCA530 readiness is `source_only_no_artifact` until an official release
   URL or files are available.
2. Implement the TelecomTS paired runner/analyzer and tune only on its source-held-out development
   and validation sessions; keep its test preregistration in draft until the algorithm is locked.
3. Use telco-v3 only to develop generic candidate verification, fault-ontology normalization, and
   evidence-falsification mechanisms; do not optimize scenario-specific rules.
4. Run the OpenRCA prereg generator after real data is present.
5. Run one-row data validation and three-row smoke for `single_react`, `single_react_sc`,
   `same_board_single`, and `shardrca_full`.
6. Freeze the final algorithm before deriving/generating telco-v4; never run v4 as a development
   smoke suite.
7. Keep RCAEval v7 result in the appendix as supporting evidence, with all leakage fixes and
   same-board limitation disclosed.

---

## 6. Completion Definition

The goal is complete only when we have:

- a real benchmark run on OpenRCA Telecom or TN-RCA530, or a documented failure to obtain both plus a
  scientifically designed telecom-valid synthetic fallback;
- preregistered systems, budgets, split/row IDs, metrics, tests, and stopping rule;
- MAS beating the strongest operational single baseline clearly on the primary metric;
- same-board oracle and equal-token/equal-budget diagnostics reported;
- mechanism ablations showing the win comes from evidence partition, graph/topology reasoning,
  adjudication, verification, or repair rather than prompt luck;
- report tables with token/cost/latency accounting and limitations.

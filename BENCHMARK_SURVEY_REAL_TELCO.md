# Real/Fallback RCA Benchmark Survey

**Date:** 2026-07-04
**Purpose:** choose benchmark evidence for a defensible claim that a multi-agent RCA system beats
strong single-agent baselines, without tuning to convenient data.

## Decision Summary

The headline benchmark should be real or operational RCA data:

1. **TN-RCA530 / TN-AutoRCA** is the best conceptual match because it is telecom alarm RCA with
   graph-structured root-cause reasoning.
2. **OpenRCA Telecom** is the best currently integrated real/operational benchmark path in this repo.
3. **OpenRCA 2.0** is newly relevant because it adds step-wise causal path annotations, but no
   local adapter/data artifact is available yet.
4. **TelecomTS** is the best immediately available public development track: measured 5G testbed
   KPIs with ten literature-grounded synthetic RCA classes. It is not real-fault RCA evidence.
5. **TeleLogsAgent** remains the preferred agentic synthetic fallback because it is telecom-specific,
   5G, and tool-use oriented, but access is gated.
6. **RCAEval-Hard** remains supporting evidence only. It is useful for engineering and label-safety
   checks, but it is microservice RCA, not telecom RCA.

Current local readiness is captured by `results/benchmark_readiness.json`: no real headline or
gated fallback benchmark data is currently ready locally. TelecomTS anomaly files are now local,
with a label-safe event-level paired runner/analyzer and draft source-session-held-out
preregistration, but readiness remains false until the algorithm is frozen. The inspected local
telco-v3 artifact remains development-only despite matching its earlier preregistration.

## Benchmark Matrix

| Benchmark | Status | Why It Matters | Local Support | Claim Role |
|---|---|---|---|---|
| TN-RCA530 / TN-AutoRCA | Source-only candidate; no official artifact URL configured/found locally | Telecom alarm RCA, knowledge-graph reasoning, long-tail/difficult cases | Planned only; readiness status `source_only_no_artifact` | Best headline if official data is obtained |
| OpenRCA Telecom | Official Google Drive download partial; quota-limited before completion | Multi-modal operational RCA over logs/metrics/traces/dependencies | Loader, evaluator, paired runner, prereg, readiness checker, downloader script | Headline candidate |
| OpenRCA 2.0 | Paper found; no local data/adapters yet | Cross-system RCA with step-wise causal annotations and process metrics | Survey only | Future headline/support if artifact is released |
| TelecomTS | Public anomaly subset downloaded; test prereg remains draft | Real 5G testbed KPIs with a published 10-class synthetic RCA task | Downloader, event merger, paired runner/analyzer, source-session split, readiness guard | Public testbed-backed synthetic development/fallback only |
| TeleLogsAgent | Gated data missing locally | Synthetic 5G tool-use benchmark for agentic troubleshooting | Loader, prereg, readiness checker | Synthetic fallback/secondary |
| TeleLogs | Gated Hugging Face dataset missing locally | Synthetic 5G drive-test RCA with realistic RF/network variables | Loader, prereg, paired runner, result analysis, downloader/readiness | Synthetic fallback context |
| Telco Troubleshooting Agentic Challenge | Gated Hugging Face/Zindi challenge; no runner yet | Wireless/IP network troubleshooting with agent tools and leaderboard scoring | Downloader/readiness only | Future telco agentic benchmark candidate |
| Synthetic telco-v3 export | Locally generated | Telecom-valid simulator export with topology, alarms, KPI/log windows, causal graph labels | Exporter, validator, readiness checker | Last-resort synthetic fallback only |
| RCAEval-Hard | Data available via existing symlink/artifacts | Multi-source RCA, useful hard split and label-safety checks | Adapter, ShardRCA, prereg/results | Appendix/supporting evidence only |

## Source Notes

### OpenRCA

Primary source: [microsoft/OpenRCA](https://github.com/microsoft/OpenRCA)

The official repository describes OpenRCA as a benchmark for assessing LLM root-cause analysis in
software operating scenarios. The task requires analyzing large telemetry volumes across KPI time
series, dependency trace graphs, and logs. The paper/search metadata reports 335 failure cases and
over 68 GB of de-identified telemetry.

Scientific implication:

- Strong fit for testing whether MAS helps when evidence is large, heterogeneous, and dependency
  structured.
- Not pure telecom alarm RCA, so the claim should say "operational RCA / OpenRCA Telecom" unless
  paired with telecom-specific evidence.
- Because data is large and expensive to run, row IDs and hashes must be preregistered before live
  LLM calls.

Repo support:

- `telco_mas/openrca/dataset.py`
- `telco_mas/openrca/cli.py`
- `telco_mas/openrca/prereg.py`
- `telco_mas/openrca/result_analysis.py`
- `telco_mas/evaluation/benchmark_readiness.py`
- `scripts/download_openrca_telecom.sh`

The local runner supports one paired result file via `--systems` or a frozen `--prereg`, so every
system is evaluated on the identical preregistered row set before paired McNemar/bootstrap analysis.

Acquisition status:

- Official folder listing exposes `Telecom.zip` with file ID
  `1cyOKpqyAP4fy-QiJ6a_cKuwR7D46zyVe`.
- A 2026-07-03 `gdown` attempt downloaded a partial 1,502,609,408-byte file to `/tmp`, then Google
  Drive returned a quota/rate-limit message.
- A 2026-07-04 retry failed before creating a partial zip with the same Google Drive quota message;
  see `results/openrca_telecom_download_attempt_2026-07-04.json`.
- Resume command is `bash scripts/download_openrca_telecom.sh --extract`.
- Readiness remains false until the zip is fully downloaded, extracted, and a frozen preregistration
  matches current query/telemetry hashes.

### OpenRCA 2.0

Primary source: [arXiv: OpenRCA 2.0](https://arxiv.org/abs/2606.27154)

The 2026-06-25 paper introduces PAVE and OpenRCA 2.0, described as a 500-instance cross-system RCA
benchmark with step-wise causal annotations and process-level metrics such as path reachability,
node F1, and edge F1. It reports that exact root-cause set recovery averages 20.7% across 11
frontier LLMs, while ungrounded diagnosis remains a visible failure mode.

Scientific implication:

- Very aligned with the mechanism we want to prove: MAS should help only if decomposition, evidence
  partitioning, and verification improve causal path grounding, not just final root-name guessing.
- If an official artifact is released, this should be added as a high-priority adapter and scored
  with causal-path metrics before making any broad RCA claim.
- Current state is survey-only; do not cite OpenRCA 2.0 as empirical evidence until data and schema
  are locally available and preregistered.

### TN-RCA530 / TN-AutoRCA

Sources:

- [arXiv: TN-AutoRCA](https://arxiv.org/html/2507.18190v1)
- [arXiv abstract](https://arxiv.org/abs/2507.18190)
- [OpenReview entry](https://openreview.net/forum?id=s5mwg63B02)
- [Hugging Face paper page](https://huggingface.co/papers/2507.18190)

The paper positions telecom RCA as graph-based and benchmark-scarce. Its TN-RCA530 benchmark is
described as public, with 530 real-world telecom alarm RCA scenarios, expert-validated KGs,
macro-F1 tuple scoring, realistic root-cause distribution, and a difficult-scenario-heavy design.

2026-07-04 source refresh:

- The arXiv HTML/abstract and OpenReview entry confirm the paper/source, but no local dataset path or
  schema is present in this workspace.
- The Hugging Face paper page currently lists no datasets citing/linking the paper.
- Web/source search did not identify an official TN-RCA530 download URL or repository that should be
  wired into this repo.
- Local readiness therefore reports `source_only_no_artifact`, not `ready`, `missing_data`, or a
  runnable benchmark.

Scientific implication:

- Best match to the user's target: telecom network alarm RCA.
- If official data is obtainable, this should supersede OpenRCA as the headline benchmark.
- Do not infer labels from alarm names or graph file paths; runtime should expose only graph/alarm
  evidence and candidate universes, while evaluator keeps root-cause tuples hidden.

Missing work:

- Official data access path is not yet resolved locally.
- No adapter exists yet because the exact released file schema is not available in the workspace.
- Do not invent a TN-RCA530 schema from the paper prose. The adapter should be implemented only after
  official files or a release URL are available.

### TelecomTS

Primary sources:

- [TelecomTS dataset](https://huggingface.co/datasets/AliMaatouk/TelecomTS)
- [TelecomTS paper](https://arxiv.org/abs/2510.06063)
- [Official code](https://github.com/Ali-maatouk/TelecomTS)

TelecomTS contains 18 PHY/MAC/network KPI channels sampled at 10 Hz from a controlled 5G testbed.
The public repository exposes 1,075 synthetic-anomaly RCA windows across ten classes and 279 real
over-the-air jamming windows. The important boundary is upstream task construction: the published
root-cause classifier explicitly skips `Jamming` and predicts only the ten anomaly types injected
into measured KPI traces. Therefore this is **testbed-backed synthetic RCA**, not real-fault RCA.

Scientific implication:

- It is more externally grounded than the local simulator because the base signals come from a live
  5G setup and the anomaly transformations are documented against telecom literature.
- It cannot replace TN-RCA530/OpenRCA for the headline, and a win must be titled synthetic-only.
- The upstream code creates a random 80/20 row split. Adjacent windows from one collection session
  can then cross train/test boundaries, so this repo instead holds out whole `zone x application`
  source sessions using a fixed Latin-square assignment.
- Upstream windows have length 128 and stride 32. Local validation found that 363 held-out windows
  represent only 39 anomaly events; treating windows as independent would understate uncertainty.
  The runner merges overlapping windows, deduplicates shared KPI points, and scores one diagnosis
  per event.
- Absolute timestamps, source paths, `anomalies`, affected-KPI labels, generated troubleshooting
  tickets, Q&A answers/reasoning, generated descriptions/statistics, and `anomaly_present` are all
  evaluator-only. Runtime receives opaque IDs, raw relative KPI arrays, benign scenario context,
  and the fixed candidate universe.

Local state:

- `scripts/download_telecomts.sh` downloads only the 12 anomaly JSONL files (nine synthetic and
  three jammer sessions), about 53 MB rather than the full 1.27 GB repository.
- `telco_mas/telecomts/dataset.py` validates the official 18 x 128 schema and loads only the ten
  upstream RCA classes.
- `telco_mas/telecomts/prereg.py` defines development/validation/test by source session and freezes
  all 39 independent test events. Two rare classes have only one test event each, so macro accuracy,
  event-level McNemar, and per-class uncertainty are mandatory.
- `telco_mas/telecomts/cli.py` gives full MAS five calls (one full-board generalist, three
  disjoint specialists, and one final adjudicator) and gives `single_equal_calls` five full-board
  calls. This is call-matched, not exactly token-matched; measured tokens are reported without
  relabeling the baseline.
- The current MAS adds deterministic domain guards over specialist outputs: radio-primary classes
  require radio-specialist consistency, resource-allocation overrides require weak radio support plus
  strong resource evidence, and aligned abrupt resource/traffic load promotes sudden congestion. A
  nine-event validation-source calibration shows MAS above the strongest single baseline, but the
  paired exact test is still not significant. Because those events were inspected while tuning the
  guard, they are development evidence rather than held-out validation; the algorithm is not locked.
- `telco_mas/telecomts/result_analysis.py` selects the strongest single by macro accuracy, then
  micro accuracy, measured tokens, and lexical name.
- A development-only shape-prototype diagnostic exists, but it is not enabled in the default runner:
  source-session validation showed that nearest-prototype matching can overfit development sessions
  instead of root-cause mechanisms.
- `results/prereg_telecomts_draft.json` is intentionally `draft`; it must not be frozen until the
  prompts, tools, systems, and algorithm ID are locked on development/validation.
- Readiness remains false and claim audit requires `--allow-synthetic` even after a future clear win.

### TeleLogsAgent

Primary source: [netop/TeleLogsAgent on Hugging Face](https://huggingface.co/datasets/netop/TeleLogsAgent)

The Hugging Face card describes TeleLogsAgent as a benchmark/evaluation framework for measuring LLM
agents' structured tool use in telecommunications. The page explicitly asks users not to publicly
share or redistribute the dataset, so this should be treated as gated benchmark data.

Scientific implication:

- Good fallback for agentic MAS because it tests tool use in telecom troubleshooting.
- Still synthetic, so it cannot be the final real-telco proof unless real benchmark access fails and
  the report states that limitation.

Repo support:

- `telco_mas/telelogs_agent/dataset.py`
- `telco_mas/telelogs_agent/prereg.py`
- `telco_mas/telelogs_agent/cli.py`
- `telco_mas/telelogs_agent/result_analysis.py`
- `telco_mas/evaluation/benchmark_readiness.py`
- `scripts/download_telelogs_agent.sh` downloads the gated Hugging Face files once the user has
  accepted the dataset conditions and provided `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`.

Current local access check:

- 2026-07-03: `scripts/download_telelogs_agent.sh` failed with Hugging Face `401` gated-repo access
  for `netop/TeleLogsAgent` because no authenticated accepted-token is available in the environment.
- 2026-07-04: retrying with a user-provided Hugging Face token failed with `403`; the account is not
  in the authorized list for the gated dataset. This is an external data-access blocker, not an
  adapter failure.

Implementation caveat: the local runner now has `--mode tool` for the official TeleLogsAgent
FastAPI endpoints, plus lower-evidence `profile` and label-safe `llm` staged modes. The tool path
records task success, tool-call volume, tool-call failure rate, and score-per-tool-call efficiency,
matching the benchmark's tool-use emphasis. No local TeleLogsAgent evidence exists yet because the
gated `TS1`/`TS2`/`TS3` files and server code are not available without accepted Hugging Face access.

### TeleLogs

Primary source: [netop/TeleLogs on Hugging Face](https://huggingface.co/datasets/netop/TeleLogs)

The dataset card describes TeleLogs as synthetic 5G RCA data built around realistic network
engineering parameters and drive-test scenarios involving UE movement through gNodeB coverage. It
is more telecom-valid than generic synthetic incident suites.

Scientific implication:

- Useful to justify fallback schema design if TeleLogsAgent cannot be run.
- Better as secondary context than primary MAS evidence because it is a text RCA benchmark rather
  than an official HTTP-tool agent benchmark.
- Local support now includes `telco_mas/telelogs/dataset.py`, `prereg.py`, `cli.py`, and
  `result_analysis.py`. Runtime payloads strip root-cause labels; strict scoring requires an exact
  structured `root_causes` set, while prose-only label mentions are partial diagnostic signal only.
- Local readiness tracks `data/telelogs` and `results/prereg_telelogs_frozen.json`; it becomes
  fallback-ready only when the gated files exist and the frozen manifest matches.

### Telco Troubleshooting Agentic Challenge

Primary sources: [Hugging Face dataset](https://huggingface.co/datasets/netop/Telco-Troubleshooting-Agentic-Challenge)
and [Zindi competition](https://zindi.africa/competitions/telco-troubleshooting-agentic-challenge)

This 2026 challenge is agentic and telecom-specific, with wireless and IP troubleshooting tracks.
Track A uses a simulation tool server for wireless tasks; Track B uses an IP network environment
with device CLI/tool interactions. Public pages describe leaderboard scoring over answer accuracy,
with an efficiency discount in the final phase.

Scientific implication:

- Strong future fit for the MAS-vs-single claim because it is explicitly agentic and tool-oriented.
- Not usable as evidence locally yet: the Hugging Face repository is gated, and a dedicated
  label-safe adapter/scorer must be written after access to the official files.

### Synthetic telco-v3 Export

Local artifact: `results/synthetic_telco_v3_dataset.json`

Frozen preregistration: `results/prereg_synthetic_telco_v3_frozen.json`

Calibration results:

- `results/synthetic_telco_v3_full_vs_single_kpi_policy_analysis.json`: negative clear-win gate
  (`diagnosis_correct` delta `-0.1667`, exact paired `p=0.6875`).
- `results/synthetic_telco_v3_full_vs_single_power_expert_analysis.json`: negative clear-win gate
  despite full-MAS improvement; full 8/12 strict, single 9/12 strict, delta `-0.0833`,
  exact paired `p=1.0`.
- `results/synthetic_telco_v3_verifier_dev_analysis.json`: still negative; full 11/12 strict,
  single 10/12 strict, delta `+0.0833`, exact paired `p=1.0`; end-to-end delta `0.0`.
  `results/synthetic_telco_v3_verifier_ablation_analysis.json` shows full and `no_verifier` tied,
  so the verifier has not yet demonstrated an independent contribution.

These are scientifically useful failure/calibration signals, not supporting evidence for the final
claim. Telco-v3 is now development-only because its outcomes have been inspected while tuning.

This is not a real benchmark and not a gated public dataset. It is a structured export of the local
telecom simulator's hard suite (`telco_v3`) for last-resort fallback use. Each case has:

- label-safe runtime payload: opaque case ID, incident description, alarms, anomalous KPIs, logs, and
  observable topology neighborhood;
- evaluator-only labels: root element, fault family, acceptable elements, remediation SOP/keywords,
  stress tags, secondary faults, causal graph, and difficulty bin;
- validation invariants: runtime must not contain evaluator-only label keys, every case must have
  alarms/KPIs/causal edges, and difficulty must be easy/middle/hard.
- readiness gate: preregistration status must be frozen, artifact SHA256 must match, and runtime
  case IDs must match in order.

Scientific implication:

- Better than ad hoc synthetic examples because it exports causal graph and telecom topology
  structure, but still weaker than OpenRCA/TN-RCA/TeleLogsAgent.
- Any result on it must be titled "synthetic telco-v3 fallback" and cannot be the final real-telco
  claim.
- The latest post-KPI-policy calibration did **not** show a MAS win; it motivated adding an explicit
  power/site expert, which requires a fresh preregistered run before any new claim.

### RCAEval

Sources:

- [RCAEval GitHub](https://github.com/phamquiluan/RCAEval)
- [RCAEval Zenodo](https://zenodo.org/records/14590730)
- [RCAEval paper](https://arxiv.org/html/2412.17015v5)

RCAEval includes 735 failure cases across Online Boutique, Sock Shop, and Train Ticket, with
multi-source telemetry and annotated root-cause service/indicator fields.

Scientific implication:

- Useful for hardening ShardRCA and checking label-safety.
- Not telecom; do not use it as the headline for the user's objective.
- Existing v7 result should be reported as supporting positive evidence for the budgeted
  single-context comparison, with the same-board oracle limitation disclosed.
- Re-analysis with `--baseline strongest_single` resolves the strongest single to
  `rcaeval_same_board_single`; MAS scores 0.60 Hit@1 vs 0.70 for that oracle-like single, so this
  bounds the mechanism claim rather than supporting a broad MAS-win statement
  (`results/rcaeval_hard_llm_v7_holdout20_analysis_strongest.json`).

## Required Evidence Before Final Claim

The goal is not scientifically complete until at least one of these is true:

1. TN-RCA530 official data is obtained, an adapter is implemented, and MAS beats the strongest
   operational single baseline under a frozen preregistration.
2. OpenRCA Telecom data is obtained, readiness passes, a preregistered run is executed, and MAS beats
   the strongest operational single baseline on strict/partial OpenRCA metrics.
3. If neither real benchmark is obtainable, TeleLogsAgent or a clearly telecom-valid synthetic
   fallback is run under frozen preregistration, with the final claim explicitly limited to synthetic
   5G troubleshooting.

## Anti-Trick Checklist

- No tuning on RCAEval v7 holdout.
- No live run unless `benchmark_readiness` says data and prereg match.
- No post-hoc row filtering.
- Same primitive tools for MAS and single baselines.
- Report same-board oracle, token/tool/latency accounting, and per-difficulty breakdown.
- Synthetic fallback must be labeled as synthetic in the title/table, not buried in a footnote.

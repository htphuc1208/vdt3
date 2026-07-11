# Infrastructure-Only Telecom Evaluation Track

This track defines what counts for the project goal: a multi-agent method must
beat a comparable single-agent method on RAN, mobile-core, or transport-network
data. Software-service and microservice benchmarks are supporting evidence only.

## Confirmatory result

The frozen `icas-spgc-inductive-v3-final` protocol was evaluated once on the
preserved 600-case ICASSP-SPGC test labels. It satisfies the project goal:

| Metric | Multi-agent | Equal-compute single | Difference |
|---|---:|---:|---:|
| Official challenge score | **0.7925** | 0.7833 | **+0.00917** |
| Macro-F1 | **0.7223** | 0.6957 | +0.0266 |
| Exact root-set accuracy | **0.6950** | 0.6833 | +0.0117 |

The primary per-case score delta has a paired-bootstrap 95% interval of
`[0.00083, 0.01833]`, excluding zero. Seven cases were exactly correct only for
the multi-agent system and none only for the single agent (exact paired
binomial/McNemar p=`0.015625`). With one preregistered primary comparison, the
Holm-adjusted p-value is unchanged and below 0.05.

This is a win over the matched single-agent comparator, not a SOTA result. The
multi-agent score is below the preserved third-place report of 0.93. Exact
leaderboard comparability is also qualified because the original challenge site
is offline and the experiment uses a contestant-preserved copy of the data and
post-competition labels. Full hashes, thresholds, resource matching, and per-root
metrics are recorded in `results/ICAS_SPGC2022_RESULT.md` and the generated JSON.

## Benchmark decision

1. **ICASSP-SPGC 2022 Wireless Network Fault Localization** is the immediately
   runnable headline benchmark. It contains 2,984 variable-length time slices
   collected from live 5G drive-test scenarios, 23 observed variables, an
   expert-endorsed causal graph, and expert root-cause labels. The official
   challenge site is offline; the official baseline code remains public and the
   third-place team's repository preserves the processed 1,407-case train and
   600-case blind-test artifacts plus post-competition labels. Its reported
   leaderboard score is 0.93.
2. **TN-RCA530 / the complete TeleCom-Bench Root Cause Diagnosis set** is the
   preferred headline task. TN-RCA530 describes 530 real operational base-station
   alarm/topology graphs and reports Auto-RCA at F1 0.9179. TeleCom-Bench describes
   983 authentic root-cause cases across wireless, wired, and core workflows.
   As of 2026-07-10, neither complete evaluation set has a public download URL:
   TeleCom-Bench publishes evaluation code and examples, and directs researchers
   to its maintainers for the complete data.
3. **SpotLight** is the public fallback. It contains real measurements from a
   commercial-grade 5G Open RAN deployment (CU/DU/RU, radio and fronthaul) with
   injected PDCP, MAC, radio, network, and mixed anomalies. It is testbed-backed,
   not production-fault data. The paper reports 0.95 overall F1.
4. **NIST DARE** is valid real RAN testbed data but is too narrow for the main
   comparison: it detects only encryption-on versus encryption-off, and the
   published single-feature detector is already near 0% false alarms and 100%
   detections.
5. **OpenRCA Telecom and RCAEval are excluded from the headline.** OpenRCA is
   explicitly a software-operating benchmark and its Telecom components are
   Docker, OS, and database nodes. RCAEval is a microservice benchmark.

Primary sources:

- IEEE challenge page: https://signalprocessingsociety.org/publications-resources/data-challenges/root-cause-analysis-wireless-network-faults-localization
- Official challenge baselines: https://github.com/zhangtj1996/RCA-telecom
- Preserved third-place artifact: https://github.com/zxuan000/SPGC_aiops_bjtu
- TN-AutoRCA: https://arxiv.org/abs/2507.18190
- TeleCom-Bench: https://github.com/ZTE-AICloud/TeleCom-Bench
- SpotLight: https://github.com/netsys-edinburgh/SpotLight
- NIST DARE: https://doi.org/10.18434/mds2-3430
- OpenRCA: https://github.com/microsoft/OpenRCA

## Integrity gates

- Runtime inputs never include `label.json`, scoring points, or evaluator fields.
- The public TeleCom-Bench example currently contains a `causedBy` edge marked
  `targetRootCause`. Raw-input scores from such graphs are rejected. The clean
  protocol removes explicit answer-marker fields before inference and reports
  that it is not directly comparable with a raw-input published score.
- SpotLight point-level F1 is not computed unless every evaluated experiment has
  an upstream ground-truth interval file. The public Drive folder currently lists
  KPI files but no complete label set. Labels must not be reverse-engineered from
  KPI values, filenames, or nominal injection timing.
- Development, validation, and test cases are separated before prompts, fusion
  weights, or decision rules are tuned. No post-hoc row filtering is allowed.

Run the current audit with:

```bash
make download-telecom-bench
make real-telecom-readiness
```

## Win criterion and realized protocol

The primary comparison uses the same base estimator, number of model bundles,
hyperparameters, and runtime KPI evidence for both systems:

- single agent: three-seed self-consistency over the complete KPI summary;
- multi-agent: level, temporal-dynamics, and causal-path KPI specialists plus an
  out-of-fold logistic adjudicator;
- primary metric: the official paired per-case challenge score;
- required result: positive effect with a 95% paired bootstrap interval excluding
  zero and Holm-adjusted paired-test p <= 0.05;
- SOTA claim: only if the exact published input and scoring protocol is reproduced
  without answer leakage.

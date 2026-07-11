# ICASSP-SPGC 2022 Confirmatory Result

This is the frozen result record for the infrastructure-only telecom track.

- Dataset: live 5G road-test wireless fault-localization data preserved by the
  challenge's third-place team
- Training cases: 1,407 (984 adjudicator-development core, 423 threshold validation)
- Held-out test cases: 600
- Protocol: `icas-spgc-inductive-v3-final`
- Protocol SHA-256: `dcd0414a20200a2203888107ac75eaa044df6e71fb7b814f5ce3c3a4c143221e`
- Preserved data commit: `d6dc5f270a20a29cfc43703cd018826f9c52e5bd`
- Confirmatory test run: 2026-07-10

## Result

| Held-out metric | Multi-agent | Equal-compute single | Delta |
|---|---:|---:|---:|
| Official challenge score | **0.792500** | 0.783333 | **+0.009167** |
| Micro-F1 | **0.824561** | 0.817874 | +0.006687 |
| Macro-F1 | **0.722259** | 0.695692 | +0.026567 |
| Exact root-set accuracy | **0.695000** | 0.683333 | +0.011667 |

Paired inference on the 600 per-case official scores:

- bootstrap 95% CI for the mean delta: `[0.000833, 0.018333]`;
- multi better / single better / tied cases: 11 / 3 / 586;
- exact-set discordances: multi-only correct 7, single-only correct 0;
- exact two-sided paired binomial/McNemar p-value: `0.015625`.

## Per-root test metrics

| Root | System | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Root1 | Multi | 0.967427 | 0.818182 | **0.886567** | 363 |
| Root1 | Single | 0.961538 | 0.757576 | 0.847458 | 363 |
| Root2 | Multi | 0.969697 | 0.275862 | **0.429530** | 116 |
| Root2 | Single | 1.000000 | 0.215517 | 0.354610 | 116 |
| Root3 | Multi | 0.982578 | 0.750000 | 0.850679 | 376 |
| Root3 | Single | 0.977492 | 0.808511 | **0.885007** | 376 |

## Resource and leakage controls

Both systems use three ExtraTrees bundles with the same 400 trees, feature
sampling, leaf size, class weighting, and seeds. The single system applies all
three bundles to the complete 336-feature summary. The multi-agent system applies
one bundle each to level (192 features), dynamics (144), and causal-path (154)
views; three small per-root logistic adjudicators consume only out-of-fold
specialist probabilities.

The preserved training table's `causes_type` field and all IDs/label columns are
deleted before feature discovery. Single thresholds `[0.49, 0.70, 0.51]` and
multi thresholds `[0.56, 0.92, 0.50]` were selected on the 423-case training-only
validation split. In the hardened runner, test-label parsing and hashing are
deferred until all fits and threshold choices are complete.

Transparency note: during artifact discovery, aggregate test-label prevalence
was inspected, and the initial readiness implementation parsed label counts
before fitting. Test labels never entered a feature, estimator, adjudicator, or
threshold-selection API, and no test prediction existed during development, but
the first evaluation was therefore not operator-blind. The hardened rerun uses
the exact already-frozen protocol and changes only label-loading order.

The machine-readable record is `results/icas_spgc2022_single_vs_multi.json`
(locally generated and git-ignored because result JSONs may contain per-case
predictions). Reproduce it with:

```bash
make download-icas-spgc
make bench-icas-spgc
```

## Claim boundary

The multi-agent system significantly beats the matched single-agent system on
this real RAN benchmark. It does not beat the preserved third-place score of
0.93, so no SOTA claim is made. The original challenge site is offline; the
leaderboard comparison therefore remains qualified even though the preserved
test protocol and scoring function are used.

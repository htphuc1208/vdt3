# RCAEval Paper Reproduction: BARO vs ShardRCA

Generated: 2026-07-05

## Sources

- RCAEval official repository: https://github.com/phamquiluan/RCAEval
- RCAEval paper: https://arxiv.org/abs/2412.17015
- BARO paper: https://arxiv.org/abs/2405.09330

## Reproduced Baseline

- Repository clone: `vendor/RCAEval-www25`
- Branch: `www25`
- Commit: `9d14687ce0644188f1f1a576fd3f57cd903af446`
- Runner wrapper: `scripts/run_official_rcaeval_baro.py`
- Baseline: official RCAEval `main.py --method baro`

Full official BARO reproduction:

| Dataset | Avg@5 CPU | MEM | DISK | SOCKET | DELAY | LOSS | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RE2-TT | 0.72 | 0.99 | 1.00 | 0.83 | 0.63 | 0.64 | matches RCAEval paper Table 6 example |
| RE2-OB | 0.64 | 0.87 | 0.81 | 0.67 | 0.67 | 0.80 | reproduced locally |
| RE2-SS | 0.80 | 0.83 | 0.77 | 0.71 | 0.68 | 0.80 | reproduced locally |

## ShardRCA Paired Pilot

- Dataset: RCAEval RE2-TT
- Paired subset: 20 cases
- Seed: `20260706`
- Excluded engineering smoke case: `RCAEval-RE2-TT-ts-travel-service_delay-1`
- ShardRCA output: `results/rcaeval_paper_re2_tt_shardrca20.json`
- Paired analysis: `results/rcaeval_paper_re2_tt_baro_vs_shardrca20_analysis.json`

| System | n | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RCAEval WWW'25 BARO | 20 | 0.55 | 0.75 | 0.80 | 0.71 | 0.6537 |
| ShardRCA full | 20 | 0.50 | 0.55 | 0.55 | 0.53 | 0.5342 |

Paired tests:

- Hit@1 McNemar: BARO-only correct = 6, ShardRCA-only correct = 5, exact p = 1.0.
- Avg@5 paired bootstrap: ShardRCA - BARO = -0.18, 95% CI [-0.41, 0.03].

Resource use for ShardRCA on 20 cases:

- Tokens: 5,533,748
- LLM calls: 678
- Tool calls: 740
- Summed per-case latency: 4,113.81 seconds

## Interpretation

This is a paired pilot, not a full paper-scale claim. BARO is stronger on ranking depth in this subset, especially Avg@5 and AC@3/AC@5. Hit@1 is close and statistically inconclusive at n=20. ShardRCA is notably expensive compared with BARO and needs either a full RE2-TT run or a cheaper ablation before claiming competitiveness.

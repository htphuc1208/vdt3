# RCAEval Metric-Prior Repair Report

Generated: 2026-07-05

## Change

ShardRCA now applies a label-safe RCAEval metric prior before returning an RCAEval prediction.

Rule:

- Rank services by finite pre/post metric shift.
- If the top service has margin `top / (top + runner_up) >= 0.625`, make it the primary root.
- If the metric margin is diffuse, keep the MAS winner primary and insert the top metric-prior roots after it for ranking depth.
- Map final fault type to RCAEval families (`cpu/mem/disk/delay/loss/socket`) from the metric attached to the selected root when possible.
- Do not fallback to the hidden RCAEval fault label when constructing inference payloads.

Changed code:

- `telco_mas/evaluation/rcaeval_adapter.py`
- `telco_mas/shardrca/runner.py`
- `telco_mas/evaluation/run_benchmark.py`
- `telco_mas/evaluation/rcaeval_paper_comparison.py`

## Replay Result on the Existing 20-Case RE2-TT Pilot

This is a checkpoint rescore. It does not make new LLM calls.

| System | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| BARO official | 0.55 | 0.75 | 0.80 | 0.71 | 0.6537 |
| ShardRCA before repair | 0.50 | 0.55 | 0.55 | 0.53 | 0.5342 |
| ShardRCA + metric prior | 0.85 | 0.85 | 0.90 | 0.86 | 0.8629 |

Paired tests for repaired ShardRCA vs BARO:

- Hit@1 McNemar: BARO-only correct = 1, ShardRCA-only correct = 7, exact p = 0.070312.
- Avg@5 paired bootstrap: ShardRCA - BARO = +0.15, 95% CI [-0.03, 0.33].

Files:

- Repaired ShardRCA rescore: `results/rcaeval_paper_re2_tt_shardrca20_metric_prior_rescore.json`
- Repaired BARO comparison: `results/rcaeval_paper_re2_tt_baro_vs_shardrca20_metric_prior_analysis.json`

## Scientific Boundary

This repair is exploratory. It is a method change informed by failure analysis on this pilot subset, so it must not be presented as a confirmatory win on RCAEval RE2-TT. The next valid step is a fresh preregistered run on a disjoint RCAEval split or the full RE2-TT set with the repaired code frozen beforehand.

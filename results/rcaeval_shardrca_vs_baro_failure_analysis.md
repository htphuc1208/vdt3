# Why ShardRCA Underperformed BARO on RCAEval RE2-TT Pilot

Generated: 2026-07-05

## Empirical Finding

Paired subset: 20 RCAEval RE2-TT cases, seed `20260706`, excluding the engineering smoke case `RCAEval-RE2-TT-ts-travel-service_delay-1`.

| System | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| RCAEval WWW'25 BARO | 0.55 | 0.75 | 0.80 | 0.71 | 0.6537 |
| ShardRCA full | 0.50 | 0.55 | 0.55 | 0.53 | 0.5342 |
| Adapter pre/post metric-shift heuristic | 0.70 | 0.75 | n/a | n/a | 0.7392 |

The surprising part is not just that BARO beats ShardRCA on Avg@5. A very simple label-safe metric-shift heuristic also beats ShardRCA on Hit@1 for this exact subset. That means the raw dataset contains enough root signal; ShardRCA is losing signal inside its multi-agent pipeline.

## Main Failure Modes

### 1. Candidate generation loses obvious metric roots

When ShardRCA is wrong, the true root is often ranked very low or missing entirely:

| Case group | Pattern |
| --- | --- |
| `ts-auth-service_delay-*` | true root falls to rank 8-10 |
| `ts-auth-service_loss-*` | true root absent from top 10 |
| `ts-train-service_*` | true root absent from top 10 in all three pilot cases |
| `ts-route-service_socket-3` | true root absent from top 10 |

This is worse than a final arbiter choosing the wrong top candidate. The upstream candidate set itself is already damaged.

### 2. MAS fusion amplifies propagated symptoms

RCAEval Train Ticket has many correlated downstream metric shifts. BARO is designed to rank metric-level root indicators under this setting. ShardRCA splits telemetry into agents, asks them to produce local candidates, then rewards cross-agent convergence. If several agents observe the same propagated symptom, the fusion layer can treat agreement on a downstream service as independent support.

This is visible in errors such as:

- true `ts-train-service`, predicted `ts-price-service`, `ts-travel2-service`, or `ts-food-map-service`
- true `ts-auth-service`, predicted `ts-order-service`, `ts-inside-payment-service`, or `ts-travel2-service`
- true `ts-route-service`, predicted `ts-order-other-mongo`

Those are plausible affected services, but not the injected root service.

### 3. The method is not aligned with RCAEval's metric-level objective

BARO directly models multivariate time-series changes and is explicitly built for microservice metric RCA. ShardRCA is a general MAS diagnostic pipeline. In this adapter, RCAEval is converted into a label-safe telemetry catalog plus summary features, then passed through miner, local candidates, peer interaction, fusion, and falsifier. That adds interpretive degrees of freedom without adding benchmark-specific signal.

The simple metric-shift sanity check shows that ShardRCA needs a stronger numerical prior, not more verbal reasoning.

### 4. Fault taxonomy is currently mismatched

ShardRCA's checkpoint `fault_type` is usually an internal reason family such as `db` or `unknown`, not RCAEval's `cpu/mem/disk/delay/loss/socket`.

In the 20-case run:

- predicted `db`: 12 cases
- predicted `unknown`: 5 cases
- predicted exact RCAEval families only rarely
- official benchmark `fault_accuracy`: 0.0

The BARO comparison above scores only root ranking, so this does not explain the root loss. But it shows the adapter is not yet benchmark-native.

### 5. ShardRCA is expensive and under-calibrated

ShardRCA used 5,533,748 tokens and 678 LLM calls for 20 cases. Confidence was not reliable enough to protect against bad predictions: several wrong predictions had moderate confidence, while low confidence did not trigger abstention or fallback to the stronger metric-shift prior.

## What This Means Scientifically

The result is not evidence that MAS is useless. It is evidence that this MAS instantiation is not yet a strong RCAEval method. Its autonomy and interaction are adding complexity, but the benchmark rewards precise numerical localization over narrative cross-agent reasoning.

Current strongest interpretation:

> ShardRCA is behaving like a general diagnostic reasoning system, while BARO is behaving like a specialized statistical root-localization method. On RCAEval RE2-TT, specialization wins.

## Priority Fixes

1. Make metric-shift/BARO-style scores first-class priors in candidate generation.
2. Require every final root candidate to survive a raw metric indicator check.
3. Penalize downstream propagated symptoms unless they temporally precede or dominate the injected-root metric.
4. Disable or reduce convergence bonuses when worker evidence is highly correlated.
5. Map ShardRCA reason families into RCAEval fault families.
6. Add an abstain/fallback rule: if MAS confidence is low or the metric prior strongly disagrees, return the metric prior or run a targeted second pass.
7. Run ablations on the same 20 cases: `no_interaction`, `no_falsifier`, `no_refinement`, metric-prior-only, and full ShardRCA.

## Claim Boundary

This is a pilot, not a full paper-scale result. Hit@1 difference between BARO and ShardRCA is small and statistically inconclusive at n=20. The more serious issue is qualitative: ShardRCA loses ranking depth and can underperform a simple metric-shift heuristic, so it needs architectural repair before a full RCAEval claim.

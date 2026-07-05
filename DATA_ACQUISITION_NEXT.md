# Data Acquisition — Verified Status & Action List (2026-07-04)

Verified today with `python3 -m telco_mas.evaluation.benchmark_readiness` and a
live OpenRCA strict-data check. This is the acquisition half of the B/C/D/E plan.

## Verified state

| Benchmark | Status | Headline? | What it unblocks |
|---|---|---|---|
| **OpenRCA Telecom** | `ready` (51 rows local, strict-data check passes) | yes, but **n=51 only** | real-operational evidence; **too small** to confirm a ≤10pp effect (needs ≥234 pairs — see power analysis) |
| **TN-RCA530 / TN-AutoRCA** | `source_only_no_artifact` | best possible | 530 real telecom-alarm cases → the only local path with enough n for a small-effect confirmatory claim |
| TeleLogsAgent | `missing_data` (HF gated; earlier 403 = account not authorized) | fallback | agentic 5G tool-use synthetic fallback |
| TeleLogs | `missing_data` (HF gated) | fallback | text 5G RCA synthetic fallback |
| Telco-Troubleshooting-Challenge | `missing_data` (gated) | fallback | future agentic telco benchmark |
| TelecomTS | `needs_frozen_prereg` (data local) | synthetic dev | dev/validation only |

## The binding constraint (why D matters)

OpenRCA Telecom is the only *ready* real benchmark, but the reviewed frozen run
failed the claim gate and is now consumed diagnostic evidence. At **n=51** the
exact paired McNemar has **~18% power** for a true +10pp effect and **~8%** at
the observed +6pp. A confirmatory claim on OpenRCA Telecom alone is statistically
impossible unless the effect is ≥ ~25pp. Therefore the real full-claim path
requires **TN-RCA530 (n=530)**, authorized TeleLogsAgent tool-mode access, future
OpenRCA 2.0 artifacts, or a fresh externally valid benchmark.

## Actions that need the user (external, cannot be automated here)

1. **TN-RCA530** — obtain an official release URL or author-provided files.
   - Sources: arXiv 2507.18190, OpenReview `s5mwg63B02`, HF `papers/2507.18190`.
   - Route: request via OpenReview author contact / paper correspondence email.
   - **Do not** reconstruct the schema from prose — adapter is written only after
     official files exist (`telco_mas.tnrca`, not yet created — correctly).
2. **Gated HF sets** (TeleLogsAgent / TeleLogs / Telco-Troubleshooting-Challenge):
   - Accept dataset terms on the dataset page **with an authorized account**
     (the token used on 2026-07-04 returned `403` — the account is not on the
     allowlist). Then export `HF_TOKEN` and run the matching `scripts/download_*.sh`.

## Actions automatable once access exists (already wired)

```bash
# TN-RCA530: only after official files/schema are available
#   -> implement telco_mas/tnrca/{dataset,prereg,cli,result_analysis}.py mirroring openrca/

# Gated fallbacks (after terms accepted + authorized token):
scripts/download_telelogs_agent.sh
python3 -m telco_mas.telelogs_agent.prereg --limit-per-set 20 --out results/prereg_telelogs_agent_frozen.json
python3 -m telco_mas.evaluation.benchmark_readiness --out results/benchmark_readiness.json

# Re-verify any newly obtained data:
python3 -m telco_mas.evaluation.benchmark_readiness --strict --out results/benchmark_readiness.json
```

## Interim recommendation

Until TN-RCA530, TeleLogsAgent tool-mode access, or OpenRCA 2.0 artifacts land,
OpenRCA 1.0 should be used for error taxonomy, heuristic-floor calibration, and
validation-only repair. RCAEval high-volume remains supporting microservice
evidence only; it cannot replace a real telecom/operational headline claim after
the post-review contamination boundary.

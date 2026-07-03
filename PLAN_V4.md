# PLAN v4 — ShardRCA: beating a strong single agent on REAL telecom RCA data

**Status:** plan only — to be implemented (e.g. via Codex). No code in this document has been applied.
**Goal (user-stated):** a multi-agent method that is *clearly* stronger than a strong single-agent
baseline, demonstrated on **real telecom data**, defensible under hostile review.

---

## 0. Executive summary

Three rounds of experiments (telco_v1, holdout, telco_v3) consistently show the persona-committee
design (3 domain experts + vote) **does not beat** a strong ReAct single agent on our simulator —
v3 smoke: single 1.00 diagnosis vs full 0.33, at 2× the cost. The root cause is now identified and
verified (§1): the simulator is a **low-entropy, fully-observable world with a global anomaly
oracle** — one `query_kpis()` call returns every anomalous KPI network-wide *including the quiet
true cause*, so search is trivial and a committee has nothing to add. No amount of scenario
tweaking fixes this without an artificial arms race.

The fix is to change the battlefield and the mechanism:

1. **Battlefield → real data at real volume** where per-task telemetry is far beyond one context
   window: **OpenRCA-Telecom** (real telecom operator telemetry; published single-agent baseline
   RCA-Agent ≈ 11.34% with Claude 3.5 — huge headroom) and **RCAEval** (735 cases, already on disk).
2. **Mechanism → things a single agent structurally CANNOT replicate at matched budget**:
   - *parallel context capacity* (N workers × K-token windows over disjoint telemetry shards —
     Chain-of-Agents, NeurIPS'24, beats both RAG and long-context single models);
   - *decorrelated sampling + vote* (self-consistency / More-Agents: gains are largest exactly in
     the 10–50% single-pass accuracy band — which is where OpenRCA sits);
   - *adversarial verification* (a falsifier agent attacking the top hypothesis — targeted
     checking is much cheaper than open search; addresses the anchoring failures OpenRCA documents).

Persona experts voting on shared data = zero information diversity ("shared-blind-spot clones",
More-Agents/TMLR). Workers over **disjoint data shards** = real information diversity. That is the
entire redesign in one sentence.

---

## 1. Why v1–v3 failed (evidence, so we do not repeat it)

| Round | Result | Lesson |
|---|---|---|
| telco_v1 (n=10, 3 runs) | single diag 0.933 ≥ full 0.867; consensus/arbiter/RAG ablations ≈ 0 effect | Task ceiling: no headroom → no possible committee gain |
| holdout+distractors | full COLLAPSES (diag 0.70, res 0.50), single stable (0.90/0.60) | Earlier multi "wins" were KB answer-leak + keyword-gate artifacts |
| telco_v3 smoke (masquerade, 85 elements, partition+debate) | single 1.00/1.00 vs full 0.667/0.333; all ablations identical; debate didn't fire on the failed case | **Oracle leak** (verified): global `query_kpis()` returns all 32 anomalies incl. the quiet true cause (`optical_rx_power=-24.5` on the fiber); true-cause alarm is on the shared board. Search cost ≈ 0 ⇒ single agent ≈ optimal; partition = pure handicap |

Structural conclusion: in a small, deterministic, fully-observable world, one competent
investigator is near-optimal and any coordination is overhead. Multi-agent advantages only exist
when at least one of these binds: **(a) data volume ≫ context window, (b) single-pass accuracy in
the middle band (variance to vote away), (c) genuinely disjoint information/tools.** Real telecom
telemetry gives us all three; our sim gives none.

**Do-not list (hard rules for the implementer):**
- Do NOT tweak the simulator further to make the single agent fail (that is benchmark rigging and
  the holdout control will expose it again).
- Do NOT report profile/heuristic numbers as agent results (RCAEval profile-smoke stays a smoke).
- Do NOT compare systems at unmatched budgets; always report tokens and a budget-matched row.
- Do NOT let ground-truth labels into any inference payload (keep `inference_payload()` pattern).

---

## 2. Method v4 — **ShardRCA** (sharded-context investigation with adversarial verification)

### 2.1 Architecture

```
                        ┌─────────────────────────────────────────────┐
 task (NL query +       │ 1. PLANNER (1 LLM call)                      │
 telemetry catalog) ───►│  reads task + data catalog (files, sizes,    │
                        │  time range) → shard plan + candidate spaces │
                        └──────────────┬──────────────────────────────┘
                                       │ shards: time-window × modality × component-group
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   2. MINER worker #1        MINER worker #2   ...    MINER worker #N   (parallel, each has its
   (metrics shard)           (logs shard)             (traces shard)     OWN context; deterministic
   pandas mining tools →     template/anomaly         graph/latency      mining tools + LLM reads
   typed findings            extraction → findings    tools → findings   only the tool OUTPUTS)
              └────────────────────────┼────────────────────────┘
                                       ▼
                        ┌─────────────────────────────────────────────┐
                        │ 3. BLACKBOARD (typed, compact)               │
                        │  Finding{component, metric/log sig, window,  │
                        │  direction, magnitude, evidence_ptr}         │
                        └──────────────┬──────────────────────────────┘
                                       ▼
                        │ 4. SYNTHESIZER ×k (k=3 independent samples   │
                        │   over the blackboard → candidate root cause │
                        │   + reason + time) → majority/weighted vote  │
                        │   (reuse verified-evidence consensus)        │
                                       ▼
                        │ 5. FALSIFIER (1 targeted pass)               │
                        │   tries to DISPROVE the top candidate with   │
                        │   2-3 cheap targeted queries; if falsified,  │
                        │   promote runner-up and re-check once        │
                                       ▼
                              final answer (component, reason, time)
```

### 2.2 Why a single agent cannot replicate this at matched budget

| Mechanism | Single-agent equivalent | Why it fails |
|---|---|---|
| N parallel worker contexts over disjoint shards | Serialize shards through ONE context | Must compress/summarize between shards → cross-shard detail lost (CoA shows this loss empirically); or context overflow (OpenRCA's documented failure mode) |
| k independent synthesizer samples + vote | Sample itself k times | It CAN — so we give the single baseline self-consistency too in one ablation (`single_sc`) to be fair; the shard mechanism must carry the win, and voting over *identical* context is weaker than voting over independently mined boards |
| Falsifier with fresh context | Self-critique in same context | Anchoring: same context = correlated errors (documented in OpenRCA failure analysis and MAST) |
| Heterogeneous models (cheap miners + strong synthesizer) | One model for everything | Cost structure: we spend the saved tokens on a stronger synthesizer at equal total cost |

### 2.3 Reuse from the existing codebase

- `telco_mas/llm.py` (client, tool loop, cache, cache-only) — unchanged.
- `telco_mas/agents/consensus.py` verified-evidence tally — becomes the synthesizer vote fuser
  (findings now come from disjoint shards, so fusion is real information aggregation).
- `telco_mas/openrca/*` (dataset/evaluator/formatter/tools) — extend, don't rewrite.
- `telco_mas/evaluation/{stats,external}.py` — Wilson/McNemar/CI plumbing as-is.
- The telco_v1/v3 sim + all current results — kept as the *controlled-conditions study* section.

### 2.4 New components (implementation units for Codex)

| # | Unit | File (new) | Notes |
|---|---|---|---|
| 1 | Telemetry catalog + windowed extraction | `telco_mas/shardrca/catalog.py` | List files/sizes/time-ranges per task; extract ±30 min windows around query time; **chunked pandas** (`chunksize=`), never load full CSVs (RAM constraint) |
| 2 | Deterministic mining tools | `telco_mas/shardrca/mining.py` | per-shard: metric z-score/spike scan, log template counting (drain-like or simple clustering), trace latency aggregation. LLM sees only compact outputs |
| 3 | Planner agent | `telco_mas/shardrca/planner.py` | outputs shard plan JSON (windows × modalities × component groups) |
| 4 | Miner worker agent | `telco_mas/shardrca/miner.py` | tool loop over its shard only; emits typed `Finding` list |
| 5 | Blackboard schema | `telco_mas/shardrca/board.py` | pydantic `Finding`, dedup/rank, compact rendering |
| 6 | Synthesizer ×k + vote | `telco_mas/shardrca/synthesizer.py` | k=3 samples (temperature 0.7 for diversity), fuse via consensus tally; answer in the benchmark's exact format |
| 7 | Falsifier | `telco_mas/shardrca/falsifier.py` | 2–3 targeted mining queries against top hypothesis; flip to runner-up if falsified |
| 8 | Single baseline (RCA-Agent replica) | `telco_mas/shardrca/single_baseline.py` | ReAct + same mining tools + full data access + SAME total token budget; plus `single_sc` variant (self-consistency k=3) for maximum fairness |
| 9 | Runner + scoring | extend `telco_mas/openrca/cli.py`, `run_benchmark.py --suite openrca_llm / rcaeval_llm` | label-safe; per-task budget accounting; resumable (per-task JSON checkpoints) |

---

## 3. Real telecom datasets (priority order)

### 3.1 PRIMARY — OpenRCA **Telecom** subset (real telecom operator telemetry)
- Repo: https://github.com/microsoft/OpenRCA (ICLR'25). 335 curated failures over 3 systems;
  **Telecom** is one of them (smallest telemetry volume of the three — good for our RAM budget).
- Download: Google Drive link in the repo README → extract into `data/openrca/` (repo expects
  `Telecom/query.csv` + `Telecom/telemetry/`; our adapter already anticipates this layout and
  `OPENRCA_DATA_DIR`). Recommended machine: ~80GB disk / 32GB RAM for the FULL benchmark — we
  only take Telecom + windowed extraction, so real usage is far lower. If local RAM is too tight,
  do catalog+window extraction once and cache extracts (`data/openrca_extracts/`).
- Task: NL query → predict root-cause **component / reason / occurrence time**; scoring is
  all-or-nothing per their protocol (candidate reasons/components are given in the prompt —
  replicate their exact protocol for comparability).
- Published baseline to beat: **RCA-Agent** (single agent + code execution), Claude 3.5 ≈ 11.34%;
  weaker prompting strategies score lower. Any statistically clear win by ShardRCA here is a real,
  citable result on real telecom data.
- Sample plan: stratified 40–60 Telecom tasks (or all Telecom tasks if count permits), 1–3 runs.

### 3.2 SECONDARY (already on disk, zero download) — RCAEval RE1/RE2/RE3
- `data/rcaeval` symlink already validated at 735 cases. Microservice (not telecom) but real
  systems + real fault injection; metrics/logs/traces per case.
- Run the SAME ShardRCA vs single comparison (`--suite rcaeval --external-mode llm`) on a
  stratified sample (n=50–100). Report Hit@1/Hit@3/MRR + CIs. This is the fastest path to a real
  result — do it FIRST while OpenRCA downloads.

### 3.3 OPTIONAL — extra telecom realism
- AIOps Challenge datasets (used by mABC; telecom operator origin; netman.aiops.org mirrors) —
  only if time permits.
- TN-AutoRCA (arXiv 2507.18190) — telecom alarm RCA benchmark; check public availability.
- Kaggle "telecom network anomaly/alarm" sets — supplementary color only, not headline.

---

## 4. Experiment design (pre-registered — write this in the report BEFORE running)

**Hypothesis H1:** On tasks whose per-task telemetry volume exceeds one context window
(OpenRCA-Telecom), ShardRCA (parallel shard miners + vote + falsifier) achieves higher
root-cause accuracy than a budget-matched single agent (RCA-Agent replica), because of parallel
context capacity and decorrelated aggregation.
**H2 (mechanism):** the accuracy gap **grows with per-task telemetry volume** — bin tasks by MB of
relevant telemetry; single agent degrades with volume, ShardRCA stays flat(ter). *This chart is
the core evidence — prioritize it.*
**H0 honesty clause:** if ShardRCA does not beat `single` and `single_sc`, we report the null and
the boundary-condition story; no post-hoc metric changes.

Systems: `shardrca_full`, `single` (RCA-Agent replica), `single_sc` (self-consistency k=3),
ablations `no_falsifier`, `no_vote` (k=1), `no_shard` (1 worker, serial — isolates parallel-context).
Stats: Wilson CIs; exact paired McNemar vs `single_sc` (the strongest fair baseline); tokens and
accuracy-per-10k-tokens; identical prompt-visible information across systems.
Runs: OpenRCA-Telecom n≥40 ×1 run minimum (×3 if budget allows); RCAEval n≥50.
Budget estimate: miners on `gpt-4o-mini`, synthesizer on a stronger model within the same total
budget; expect ~60–120k tokens/task for ShardRCA ⇒ ~5–10M tokens for the headline table
(few USD on 4o-mini-class pricing; declare model/provider/temperature in meta).

---

## 5. Sim track (demoted, optional)

Keep telco_v1/v3 as the *controlled-conditions* section of the report ("when does MAS help?").
If touched at all, only to close the oracle honestly — pagination + background benign anomalies +
alarm floods (`query_kpis` never returns a global all-anomaly list; logs are voluminous) — and
re-run once. Do NOT iterate further chasing a sim win; the headline is the real-data track.

## 6. Report reframe (final narrative)

1. Controlled study (sim): strong single agents are near-optimal in small fully-observable worlds;
   persona committees add cost, not accuracy (v1–v3 + ablations + holdout — already done, honest).
2. Mechanism-grounded redesign: ShardRCA — multi-agent as *parallel context + decorrelated votes +
   adversarial verification*, aligned with CoA/More-Agents/MAST/mABC findings.
3. Real-data results: OpenRCA-Telecom (+ RCAEval) vs published single-agent baseline and our
   budget-matched `single`/`single_sc`, with the volume-scaling chart as the causal story.
4. Threats to validity carried over; pre-registered hypotheses; all artifacts reproducible.

## 7. Execution order for Codex (each step = runnable + tested before the next)

1. `catalog.py` + `mining.py` + windowed extraction with chunked IO; unit tests on RCAEval cases
   (already local). **Gate: mining tools return correct top anomalies on 3 hand-checked cases.**
2. `board.py` + `miner.py` + `planner.py`; smoke on 2 RCAEval cases with live LLM (~10 calls).
3. `synthesizer.py` (k-vote, reuse consensus tally) + `falsifier.py`.
4. `single_baseline.py` (RCA-Agent replica + `single_sc`), budget accounting.
5. RCAEval LLM run (n=50 stratified): `--suite rcaeval --external-mode llm
   --systems shardrca_full,single,single_sc` → first real table.
6. OpenRCA Telecom download + extraction cache + protocol-exact scoring; run n≥40.
7. Ablations (`no_shard`, `no_vote`, `no_falsifier`) on the cheaper of the two suites.
8. Volume-scaling analysis + charts; update `report/report.md` per §6.

Checkpoint discipline: write per-task result JSONs (resumable); enable the LLM cache for re-scoring;
never overwrite raw result files (new filename per run config, as `_default_output_path` does).

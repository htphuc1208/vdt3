# ShardRCA / OpenRCA research artifact tasks.

PYTHON ?= python3
ARGS ?=
BENCH_ARGS ?= --suite rcaeval_hard --sample 20 --systems shardrca_full,single_react_sc

.PHONY: help install install-research venv test bench bench-rcaeval-fresh bench-rcaeval-win bench-openrca prepare-openrca prereg-openrca bench-openrca-full readiness real-telecom-readiness bench-telecom-graph bench-icas-spgc download-spotlight download-telecom-bench download-icas-spgc claim-audit demo-ui clean

help:
	@echo "Targets:"
	@echo "  make install          Install lightweight runtime/test dependencies"
	@echo "  make install-research Install extra dependencies for RCAEval/OpenRCA"
	@echo "  make venv             Create .venv and install dependencies"
	@echo "  make test             Run the test suite"
	@echo "  make bench            Run run_benchmark with ARGS (defaults to RCAEval-hard profile smoke)"
	@echo "  make bench-rcaeval-fresh  Reproduce the fresh RCAEval confirmatory run"
	@echo "  make bench-rcaeval-win    Reproduce repaired MAS vs single-agent analysis"
	@echo "  make bench-openrca    Run the OpenRCA CLI"
	@echo "  make prepare-openrca  Build the immutable OpenRCA prepared cache"
	@echo "  make prereg-openrca   Freeze the current OpenRCA protocol"
	@echo "  make bench-openrca-full Resume the frozen full OpenRCA run"
	@echo "  make readiness        Rebuild benchmark readiness report"
	@echo "  make real-telecom-readiness Audit infrastructure-only telecom benchmark data and label safety"
	@echo "  make download-spotlight Download the public SpotLight 5G Open RAN data"
	@echo "  make download-telecom-bench Clone the public TeleCom-Bench examples"
	@echo "  make bench-telecom-graph Run clean-input single vs multi alarm-graph RCA"
	@echo "  make download-icas-spgc Download the ICASSP-SPGC 2022 live-5G challenge artifact"
	@echo "  make bench-icas-spgc   Run frozen single vs multi-agent live-5G evaluation"
	@echo "  make claim-audit      Rebuild claim audit report"
	@echo "  make demo-ui          Run the live ShardRCA telecom RCA demo UI"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-research:
	$(PYTHON) -m pip install -r requirements-research.txt

venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate with: source .venv/bin/activate"

test:
	$(PYTHON) -m pytest -q

bench:
	$(PYTHON) -m telco_mas.evaluation.run_benchmark $(BENCH_ARGS) $(ARGS)

bench-rcaeval-fresh:
	$(PYTHON) -m telco_mas.evaluation.run_confirmatory_fresh \
		--out results/rcaeval_hard_llm_fresh_confirm24.json $(ARGS)

bench-rcaeval-win:
	$(PYTHON) -m telco_mas.evaluation.run_group_a \
		--treatment shardrca_llmboard \
		--systems shardrca_llmboard,single_react_sc,single_equal_tokens,no_shard \
		--prior off \
		--prereg results/prereg_group_a_frozen.json \
		--checkpoint-dir results/checkpoints/group_a \
		--out results/group_a_holistic_vs_single.json $(ARGS)

bench-openrca:
	$(PYTHON) -m telco_mas.openrca.cli $(ARGS)

prepare-openrca:
	$(PYTHON) -m telco_mas.openrca.prepared --out data/openrca_prepared/Telecom $(ARGS)

prereg-openrca:
	TELCO_TEMPERATURE=0 $(PYTHON) -m telco_mas.openrca.prereg \
		--prepared-dir data/openrca_prepared/Telecom \
		--contaminated-row-ids all \
		--temperature 0 \
		--out results/prereg_openrca_telecom_frozen.json $(ARGS)

bench-openrca-full:
	TELCO_TEMPERATURE=0 $(PYTHON) -m telco_mas.openrca.cli \
		--mode llm \
		--confirm-live-llm \
		--prereg results/prereg_openrca_telecom_frozen.json \
		--prepared-dir data/openrca_prepared/Telecom \
		--checkpoint-dir results/checkpoints/openrca_telecom_frozen \
		--resume --no-cache \
		--out results/openrca_paired_frozen.json $(ARGS)

readiness:
	$(PYTHON) -m telco_mas.evaluation.benchmark_readiness --out results/benchmark_readiness.json $(ARGS)

real-telecom-readiness:
	$(PYTHON) -m telco_mas.icas_spgc.readiness \
		--root data/icas_spgc2022/SPGC_aiops_bjtu/all_file \
		--out results/icas_spgc2022_readiness.json
	$(PYTHON) -m telco_mas.spotlight.readiness \
		--root data/spotlight --out results/spotlight_readiness.json || test $$? -eq 2
	$(PYTHON) -m telco_mas.tnrca.readiness \
		--root data/telecom_bench/TeleCom-Bench/datasets/Knowledge_Application/Root_Cause_Diagnosis \
		--out results/tnrca_public_readiness.json || test $$? -eq 2

download-spotlight:
	bash scripts/download_spotlight.sh

download-telecom-bench:
	bash scripts/download_telecom_bench.sh

download-icas-spgc:
	bash scripts/download_icas_spgc2022.sh

bench-telecom-graph:
	$(PYTHON) -m telco_mas.tnrca.cli \
		--root data/telecom_bench/TeleCom-Bench/datasets/Knowledge_Application/Root_Cause_Diagnosis \
		--confirm-live-llm --out results/tnrca_paired_clean.json $(ARGS)

bench-icas-spgc:
	$(PYTHON) -m telco_mas.icas_spgc.runner \
		--root data/icas_spgc2022/SPGC_aiops_bjtu/all_file \
		--out results/icas_spgc2022_single_vs_multi.json $(ARGS)

claim-audit:
	$(PYTHON) -m telco_mas.evaluation.claim_audit --out results/claim_audit_after_repair.json $(ARGS)

demo-ui:
	$(PYTHON) scripts/demo_shardrca_live.py $(ARGS)

clean:
	rm -rf .llm_cache __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

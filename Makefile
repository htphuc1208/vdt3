# ShardRCA / OpenRCA research artifact tasks.

PYTHON ?= python3
ARGS ?=
BENCH_ARGS ?= --suite rcaeval_hard --sample 20 --systems shardrca_full,single_react_sc,same_board_single

.PHONY: help install install-research venv test bench bench-rcaeval-fresh bench-openrca prepare-openrca prereg-openrca bench-openrca-full readiness claim-audit clean

help:
	@echo "Targets:"
	@echo "  make install          Install lightweight runtime/test dependencies"
	@echo "  make install-research Install extra dependencies for RCAEval/OpenRCA/TelecomTS"
	@echo "  make venv             Create .venv and install dependencies"
	@echo "  make test             Run the test suite"
	@echo "  make bench            Run run_benchmark with ARGS (defaults to RCAEval-hard profile smoke)"
	@echo "  make bench-rcaeval-fresh  Reproduce the fresh RCAEval confirmatory run"
	@echo "  make bench-openrca    Run the OpenRCA CLI"
	@echo "  make prepare-openrca  Build the immutable OpenRCA prepared cache"
	@echo "  make prereg-openrca   Freeze the current OpenRCA protocol"
	@echo "  make bench-openrca-full Resume the frozen full OpenRCA run"
	@echo "  make readiness        Rebuild benchmark readiness report"
	@echo "  make claim-audit      Rebuild claim audit report"

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
	SHARDRCA_WEIGHTS=results/weights/local_fusion_fit_v3_temporal.json \
	$(PYTHON) -m telco_mas.evaluation.run_confirmatory_fresh \
		--out results/rcaeval_hard_llm_fresh_confirm24.json $(ARGS)

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

claim-audit:
	$(PYTHON) -m telco_mas.evaluation.claim_audit --out results/claim_audit_after_repair.json $(ARGS)

clean:
	rm -rf .llm_cache __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

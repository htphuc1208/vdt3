# TelcoMAS — one-command tasks.
# Usage: make install | make test | make demo | make bench | make dashboard

PYTHON ?= python
SCENARIO ?= fiber_cut
ARGS ?=

.PHONY: help install venv test demo list bench dashboard diagram docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  make install      Install dependencies into the current interpreter"
	@echo "  make venv         Create .venv and install dependencies there"
	@echo "  make test         Run the test suite (no API key needed)"
	@echo "  make list         List available incident scenarios"
	@echo "  make demo         Run one scenario end-to-end (SCENARIO=fiber_cut)"
	@echo "  make bench        Run the multi-agent vs single-agent benchmark"
	@echo "  make dashboard    Launch the Streamlit dashboard"
	@echo "  make diagram      Regenerate the architecture diagram"
	@echo "  make docker-build / make docker-run"

install:
	$(PYTHON) -m pip install -r requirements.txt

venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate with: source .venv/bin/activate"

test:
	$(PYTHON) -m pytest -q

list:
	$(PYTHON) -m apps.cli --list

demo:
	$(PYTHON) -m apps.cli --scenario $(SCENARIO) --trace

bench:
	$(PYTHON) -m telco_mas.evaluation.run_benchmark $(ARGS)

dashboard:
	$(PYTHON) -m streamlit run apps/dashboard.py

diagram:
	$(PYTHON) scripts/make_architecture.py

docker-build:
	docker build -t telco-mas .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env telco-mas

clean:
	rm -rf .llm_cache __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

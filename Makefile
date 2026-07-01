.PHONY: help install dev sync lock test lint format pre-commit smoke notebook-test clean demo-shapes demo-portrait demo-meshes demo-gaussian demo-all notebook portraits

UV ?= uv
NOTEBOOK := notebooks/EA Convolution.ipynb
SMOKE_OUTPUT := output/smoke

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install runtime deps (uv sync)
	$(UV) sync --no-dev

dev: ## Install project + dev deps
	$(UV) sync --dev

sync: dev ## Alias for dev

lock: ## Regenerate uv.lock
	$(UV) lock

test: dev ## Run pytest
	$(UV) run pytest

lint: dev ## Ruff check
	$(UV) run ruff check .

format: dev ## Ruff format + fix
	$(UV) run ruff format .
	$(UV) run ruff check . --fix --exit-zero

pre-commit: dev ## Install and run pre-commit on all files
	$(UV) run pre-commit install
	$(UV) run pre-commit run --all-files

portraits: dev ## Prepare Monge/Kantorovich portrait assets
	$(UV) run python -m scripts.prepare_portraits

demo-shapes: dev
	$(UV) run convw2 shapes --output $(SMOKE_OUTPUT)

demo-portrait: dev
	$(UV) run convw2 portrait --output $(SMOKE_OUTPUT)

demo-meshes: dev
	$(UV) run convw2 meshes --output $(SMOKE_OUTPUT)

demo-gaussian: dev
	$(UV) run convw2 gaussian --output $(SMOKE_OUTPUT)

demo-all: portraits demo-shapes demo-portrait demo-meshes demo-gaussian ## Run every CLI demo

notebook-test: dev ## Execute the tutorial notebook headlessly (writes to output/smoke/)
	MPLBACKEND=Agg $(UV) run jupyter nbconvert --execute --to notebook \
		--output-dir $(SMOKE_OUTPUT) --output notebook-executed.ipynb "$(NOTEBOOK)" \
		--ExecutePreprocessor.timeout=900

notebook: dev ## Launch Jupyter Lab on the tutorial notebook
	$(UV) run jupyter lab "$(NOTEBOOK)"

smoke: pre-commit test demo-all notebook-test ## Full local validation

clean: ## Remove build artefacts and venv
	rm -rf output/ build/ dist/ *.egg-info .eggs .ruff_cache .pytest_cache __pycache__ \
		src/convolutional_wasserstein/__pycache__ scripts/__pycache__ tests/__pycache__ .venv

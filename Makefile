.PHONY: help install dev demo-2d demo-3d demo-mesh notebook format lint clean

PYTHON ?= python3.10
VENV   := venv
BIN    := $(VENV)/bin

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -U pip setuptools wheel

install: $(VENV)/bin/python ## Install runtime deps in ./venv
	$(BIN)/python -m pip install -e .

dev: $(VENV)/bin/python ## Install runtime + dev/notebook deps
	$(BIN)/python -m pip install -e ".[dev,notebook]"

demo-2d: ## 2-D shape barycenter demo (writes to output/)
	$(BIN)/python -m main shapes

demo-portrait: ## 2-D portrait interpolation demo
	$(BIN)/python -m main portrait

demo-3d: ## 3-D mesh barycenter demo (slow — Monte-Carlo voxelization)
	$(BIN)/python -m main meshes

demo-gaussian: ## Heat-kernel Gaussians + barycenter on a surface mesh
	$(BIN)/python -m main gaussian

notebook: ## Launch Jupyter on the demo notebook
	$(BIN)/jupyter notebook "EA Convolution.ipynb"

format: ## Auto-format with ruff
	$(BIN)/ruff format .
	$(BIN)/ruff check . --fix --exit-zero

lint: ## Lint with ruff
	$(BIN)/ruff check .

clean: ## Remove build artefacts and cached outputs
	rm -rf output/ build/ dist/ *.egg-info .ruff_cache .pytest_cache __pycache__ src/__pycache__

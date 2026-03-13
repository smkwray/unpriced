PROJECT_VENV_ROOT ?=
ifeq ($(PROJECT_VENV_ROOT),)
PYTHON ?= python
PIP ?= pip
else
PYTHON ?= $(PROJECT_VENV_ROOT)/bin/python
PIP ?= $(PROJECT_VENV_ROOT)/bin/pip
endif
PROJECT_SLUG ?= unpaidwork
export PYTHONDONTWRITEBYTECODE = 1
export PYTHONPYCACHEPREFIX = /tmp/$(PROJECT_SLUG)-pycache
export UV_CACHE_DIR = /tmp/uv-cache-$(PROJECT_SLUG)
export RUFF_CACHE_DIR = /tmp/ruff-cache-$(PROJECT_SLUG)

.PHONY: bootstrap pull-core build-childcare fit-childcare simulate-childcare build-home fit-home report test

bootstrap:
	$(PIP) install -e '.[dev]'
	$(PYTHON) -m unpaidwork.cli bootstrap

pull-core:
	$(PYTHON) -m unpaidwork.cli pull-core --sample

build-childcare:
	$(PYTHON) -m unpaidwork.cli build-childcare

fit-childcare:
	$(PYTHON) -m unpaidwork.cli fit-childcare

simulate-childcare:
	$(PYTHON) -m unpaidwork.cli simulate-childcare

build-home:
	$(PYTHON) -m unpaidwork.cli build-home

fit-home:
	$(PYTHON) -m unpaidwork.cli fit-home

report:
	$(PYTHON) -m unpaidwork.cli report

test:
	$(PYTHON) -B -m pytest -q -p no:cacheprovider

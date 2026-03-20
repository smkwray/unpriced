PROJECT_VENV_ROOT ?=
ifeq ($(PROJECT_VENV_ROOT),)
PYTHON ?= python
PIP ?= pip
else
PYTHON ?= $(PROJECT_VENV_ROOT)/bin/python
PIP ?= $(PROJECT_VENV_ROOT)/bin/pip
endif
PROJECT_SLUG ?= unpriced
export PYTHONDONTWRITEBYTECODE = 1
export PYTHONPYCACHEPREFIX = /tmp/$(PROJECT_SLUG)-pycache
export UV_CACHE_DIR = /tmp/uv-cache-$(PROJECT_SLUG)
export RUFF_CACHE_DIR = /tmp/ruff-cache-$(PROJECT_SLUG)

.PHONY: bootstrap pull-core build-childcare fit-childcare simulate-childcare build-home fit-home report test

bootstrap:
	$(PIP) install -e '.[dev]'
	$(PYTHON) -m unpriced.cli bootstrap

pull-core:
	$(PYTHON) -m unpriced.cli pull-core --sample

build-childcare:
	$(PYTHON) -m unpriced.cli build-childcare

fit-childcare:
	$(PYTHON) -m unpriced.cli fit-childcare

simulate-childcare:
	$(PYTHON) -m unpriced.cli simulate-childcare

build-home:
	$(PYTHON) -m unpriced.cli build-home

fit-home:
	$(PYTHON) -m unpriced.cli fit-home

report:
	$(PYTHON) -m unpriced.cli report

test:
	$(PYTHON) -B -m pytest -q -p no:cacheprovider

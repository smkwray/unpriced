PROJECT_VENV_ROOT ?=
ifeq ($(PROJECT_VENV_ROOT),)
PYTHON ?= python
PIP ?= pip
else
PYTHON ?= $(PROJECT_VENV_ROOT)/bin/python
PIP ?= $(PROJECT_VENV_ROOT)/bin/pip
endif
PROJECT_SLUG ?= unpriced
CACHE_ROOT ?= $(if $(PROJECT_VENV_ROOT),$(abspath $(PROJECT_VENV_ROOT)/../.cache/$(PROJECT_SLUG)),$(HOME)/venvs/.cache/$(PROJECT_SLUG))
export PYTHONDONTWRITEBYTECODE = 1
export PYTHONPYCACHEPREFIX = $(CACHE_ROOT)/pycache
export UV_CACHE_DIR = $(CACHE_ROOT)/uv
export RUFF_CACHE_DIR = $(CACHE_ROOT)/ruff

.PHONY: bootstrap pull-core build-childcare fit-childcare simulate-childcare build-home fit-home report test

bootstrap:
	$(PIP) install -e '.[dev]'
	$(PYTHON) -B -m unpriced.cli bootstrap

pull-core:
	$(PYTHON) -B -m unpriced.cli pull-core --sample

build-childcare:
	$(PYTHON) -B -m unpriced.cli build-childcare

fit-childcare:
	$(PYTHON) -B -m unpriced.cli fit-childcare

simulate-childcare:
	$(PYTHON) -B -m unpriced.cli simulate-childcare

build-home:
	$(PYTHON) -B -m unpriced.cli build-home

fit-home:
	$(PYTHON) -B -m unpriced.cli fit-home

report:
	$(PYTHON) -B -m unpriced.cli report

test:
	$(PYTHON) -B -m pytest -q -p no:cacheprovider

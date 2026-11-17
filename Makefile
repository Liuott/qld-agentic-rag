.PHONY: help install install-eval ingest ask serve test lint format typecheck clean

PY ?= python

help:
	@echo "install      install the package with serving + dev extras"
	@echo "install-eval install the evaluation extras (ragas, datasets, pandas)"
	@echo "ingest       rebuild the Chroma index from data/sources"
	@echo "ask          interactive agent (QUERY=... for one shot)"
	@echo "serve        run the FastAPI server"
	@echo "test         run the test suite"
	@echo "lint         ruff check"
	@echo "format       ruff format + ruff check --fix"
	@echo "typecheck    mypy src"
	@echo "clean        remove caches and build artefacts"

install:
	$(PY) -m pip install -e ".[serve,dev]"

install-eval:
	$(PY) -m pip install -e ".[eval]"

ingest:
	$(PY) -m src.ingest --reset

ask:
	$(PY) -m src.app_graph $(if $(QUERY),--question "$(QUERY)",)

serve:
	uvicorn serve_api:app --host 127.0.0.1 --port 8008 --reload

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check . --fix

typecheck:
	$(PY) -m mypy src

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

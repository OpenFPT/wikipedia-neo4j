.PHONY: help install check lint typecheck test docs run weekly-slides weekly-slides-pdf

help:
	@echo "Available targets:"
	@echo "  install   - install dependencies"
	@echo "  run       - run API server"
	@echo "  lint      - run ruff"
	@echo "  typecheck - run mypy"
	@echo "  test      - run pytest"
	@echo "  docs      - build docs"
	@echo "  check     - run lint + typecheck + test"

install:
	uv sync --all-groups

run:
	uv run uvicorn src.main:app --reload --port 8000

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest

docs:
	uv run mkdocs build

check: lint typecheck test

weekly-slides:
	uv run python scripts/gen_weekly_slides.py

weekly-slides-pdf:
	uv run python scripts/gen_weekly_slides.py --pdf

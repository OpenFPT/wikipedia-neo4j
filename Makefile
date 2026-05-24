.PHONY: help install check lint typecheck test docs run up down logs schema download ingest demo weekly-slides weekly-slides-pdf

help:
	@echo "Available targets:"
	@echo "  install   - install dependencies"
	@echo "  run       - run API server"
	@echo "  demo      - run Gradio demo (port 7860)"
	@echo "  lint      - run ruff"
	@echo "  typecheck - run mypy"
	@echo "  test      - run pytest"
	@echo "  docs      - build docs"
	@echo "  check     - run lint + typecheck + test"
	@echo "  up        - start Docker services (Neo4j + Qdrant)"
	@echo "  down      - stop Docker services"
	@echo "  logs      - tail Docker service logs"
	@echo "  schema    - setup Neo4j schema"
	@echo "  download  - download Vietnamese Wikipedia dump"
	@echo "  ingest    - run ingestion pipeline"

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

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

schema:
	uv run python -m scripts.setup_neo4j_schema

download:
	uv run python -m scripts.download_viwiki

ingest:
	uv run python -m scripts.run_ingestion

demo:
	uv run python -m src.app_gradio

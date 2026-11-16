# On Windows without `make`, run the underlying `uv` commands directly (see README).
.PHONY: setup lint format test check

setup:
	uv sync --extra gpu

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

test:
	uv run pytest

check:
	uv run python scripts/check_env.py
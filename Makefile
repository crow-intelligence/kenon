.PHONY: ci format lint typecheck test

ci: format lint typecheck test

format:
	uv run ruff format --check src tests

lint:
	uv run ruff check src tests

typecheck:
	uv run ty check src

test:
	uv run pytest --doctest-modules --cov=kenon --cov-report=term-missing

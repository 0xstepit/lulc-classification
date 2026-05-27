.PHONY: unit-tests check format

check:
	uv run ruff check .

format:
	uv run ruff format .

unit-tests:
	uv run pytest ./tests/ -v

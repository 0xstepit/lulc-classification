.PHONY: unit-tests check format

check:
	uv run ruff check .

format:
	uv run ruff format .

unit-tests:
	uv run pytest ./tests/ -v

run-script:
	@echo "==================================================================="
	@echo 'Running "$(script)" script...'
	@PYTHONPATH=. uv run python scripts/$(script).py
	@echo "Completed script execution"
	@echo "==================================================================="

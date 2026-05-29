.PHONY: unit-tests check format

venv = $(pwd)/.venv
kernel_name = "lulc-classification-with-unet"

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

create-notebook-kernel:
	@echo "==================================================================="
	@echo 'Creating notebook kernel named $(kernel_name)...'
	@uv run ipython kernel install --user --env VIRTUAL_ENV $(venv) --name=$(kernel_name)
	@echo "Kernel created successfully"

start-notebook:
	@echo "==================================================================="
	@echo "Starting notebook..."
	@uv run jupyter-lab

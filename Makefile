.PHONY: unit-tests check format run-script create-notebook-kernel start-notebook

venv = $(CURDIR)/.venv
kernel_name = lulc-classification

format:
	@echo "==================================================================="
	@echo 'Running formatting...'
	uv run ruff format .
	@echo "Completed formatting"
	@echo "==================================================================="

unit-tests:
	@echo "==================================================================="
	@echo 'Running unit tests...'
	uv run pytest ./tests/ -v
	@echo "Completed unit tests execution"
	@echo "==================================================================="

run-script:
ifndef script
	$(error Usage: make run-script script=<script_name>)
endif
	@echo "==================================================================="
	@echo 'Running "$(script)" script...'
	@PYTHONPATH=. uv run python scripts/$(script).py
	@echo "Completed script execution"
	@echo "==================================================================="

kernel:
	@echo "==================================================================="
	@echo 'Creating notebook kernel named $(kernel_name)...'
	@uv run ipython kernel install --user --env VIRTUAL_ENV $(venv) --name=$(kernel_name)
	@echo "Kernel created successfully"
	@echo "==================================================================="

start-notebook:
	@echo "==================================================================="
	@echo "Starting notebook..."
	@uv run jupyter-lab

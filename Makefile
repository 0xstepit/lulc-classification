.PHONY: help install unit-tests check format list-scripts run-script
.PHONY: kernel start-notebook

venv = $(CURDIR)/.venv
kernel_name = lulc-classification
SCRIPTS_DIR = $(CURDIR)/scripts
SCRIPTS := $(notdir $(wildcard $(SCRIPTS_DIR)/*))

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "==================================================================="
	@echo 'Installing deps in a virtual environment...'
	@uv sync
	@echo "Completed installation"
	@echo "==================================================================="

format: ## Format codebase
	@echo "==================================================================="
	@echo 'Running formatting...'
	uv run ruff format .
	@echo "Completed formatting"
	@echo "==================================================================="

unit-tests: ## Run unit tests
	@echo "==================================================================="
	@echo 'Running unit tests...'
	uv run pytest ./tests/ -v
	@echo "Completed unit tests execution"
	@echo "==================================================================="

# @echo $(SCRIPTS) | sed 's/ /\n    /g'
list-scripts: ## Print available script files
	@echo $(SCRIPTS) | tr ' ' '\n'

run-script: ## Run the specified workflow script
ifndef file
	@echo "Usage: make run-script file=<script_name>"
	@echo "Specify one of the available script files:"
	@echo "$(SCRIPTS)" | tr ' ' '\n'
	exit 1
endif
	@echo "==================================================================="
	@echo 'Running "$(file)" script...'
	@PYTHONPATH=. uv run python scripts/$(file).py
	@echo "Completed script execution"
	@echo "==================================================================="

kernel: ## Create IPython kernel for the project
	@echo "==================================================================="
	@echo 'Creating notebook kernel named $(kernel_name)...'
	@uv run ipython kernel install --user --env VIRTUAL_ENV $(venv) --name=$(kernel_name)
	@echo "Kernel created successfully"
	@echo "==================================================================="

start-notebook: ## Start the Jupyter Notebook server
	@echo "==================================================================="
	@echo "Starting notebook..."
	@uv run jupyter-lab

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ium-long-stay-patterns
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
IMAGE_NAME = $(PROJECT_NAME)-predictor

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	poetry install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ium_long_stay_patterns
	isort --check --diff ium_long_stay_patterns
	black --check ium_long_stay_patterns

## Format source code with black
.PHONY: format
format:
	isort ium_long_stay_patterns
	black ium_long_stay_patterns



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created. Activate with: "
	@echo '$$(poetry env activate)'
	@echo ">>> Or run commands with:\npoetry run <command>"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	poetry run $(PYTHON_INTERPRETER) ium_long_stay_patterns/dataset.py


## Create listing stats CSV
.PHONY: listing_stats
listing_stats: data
	poetry run $(PYTHON_INTERPRETER) -m ium_long_stay_patterns.dataset --listing-stats


## service
.PHONY: run
run:
	docker build -f prediction_service/Dockerfile -t $(IMAGE_NAME):latest .
	docker run --rm -p 5000:5000 --name $(IMAGE_NAME) $(IMAGE_NAME):latest

.PHONY: dev
dev:
	poetry run python prediction_service/app.py

.PHONY: ab
ab:
	poetry run python prediction_service/ab_test.py --seed 42

.PHONY: analyze
analyze:
	poetry run python prediction_service/analyze_logs.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

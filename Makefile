#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ium-long-stay-patterns
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

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
	poetry run $(PYTHON_INTERPRETER) -c "from ium_long_stay_patterns.src.helpers.create_listing_stats_dataset import save_listing_stats; save_listing_stats()"


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

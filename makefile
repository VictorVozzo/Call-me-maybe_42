# **************************************************************************** #
#                                                                              #
#                                                          :::      ::::::::   #
#   makefile                                             :+:      :+:    :+:   #
#                                                      +:+ +:+         +:+     #
#   By: vivozzo- <vivozzo-@student.42porto.com>      +#+  +:+       +#+        #
#                                                  +#+#+#+#+#+   +#+           #
#   Created: 2026/10/22 10:45:51 by vivozzo-            #+#    #+#             #
#   Updated: 2026/11/05 13:41:34 by vivozzo-           ###   ########.fr       #
#                                                                              #
# **************************************************************************** #


SOURCE_DIR = src
FUNCTIONS = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/function_calling_results.json

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
HF_HOME := $(PROJECT_DIR)/.cache/huggingface

export PATH := $(HOME)/.local/bin:$(PATH)
export HF_HOME

all: install

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "uv is installed ($$(uv --version))"; \
	else \
		echo "uv not found. Lets install it"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync --no-cache

run:
	@uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	@uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	@uv run flake8 $(SOURCE_DIR)
	@uv run mypy $(SOURCE_DIR) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run flake8 $(SOURCE_DIR)
	@uv run mypy --strict $(SOURCE_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -rf .venv
	rm -rf .cache

re: fclean all

.PHONY: all install run debug lint lint-strict clean fclean re
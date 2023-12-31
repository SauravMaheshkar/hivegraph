## Install Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r .devcontainer/requirements.txt
	pre-commit install

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "__pycache__" -delete
	rm -rf docs/_build/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf wandb/
	rm -rf artifacts/

## Testing
test:
	pytest --durations=0 -vv .

## Basic linting
lint:
	black hivegraph
	ruff check hivegraph
	mypy hivegraph

.PHONY: lint test typecheck

lint:
	python -m ruff check --fix .
	python -m ruff format .

test:
	python -m pytest -q

typecheck:
	python -m mypy src/tinygpt --strict

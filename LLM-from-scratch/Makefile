.PHONY: run lint format clean

run:
	python3 src/main.py

format: 
	uv run ruff format .

clean:
	rm -rf .venv

lint: 
	uv run ruff check . --fix

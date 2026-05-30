.PHONY: install lint format typecheck test check train run-api run-app clean

install:        ## Sync env from lockfile
	uv sync

lint:           ## Ruff lint
	uv run ruff check .

format:         ## Ruff format
	uv run ruff format .

typecheck:      ## Pyright
	uv run pyright

test:           ## Pytest + coverage
	uv run pytest

check: lint typecheck test  ## Full local quality gate

train:          ## Fetch data, run walk-forward validation, fit + save final model
	uv run python -m market_forecast.train

run-api:        ## Serve FastAPI locally
	uv run uvicorn market_forecast.api:app --reload

run-app:        ## Serve Streamlit locally
	uv run streamlit run src/market_forecast/app.py

clean:
	rm -rf .pytest_cache .ruff_cache .pyright .coverage htmlcov

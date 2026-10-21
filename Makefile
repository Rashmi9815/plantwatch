.PHONY: install lint fmt test up down replay demo clean

install:  ## install package + dev tools and git hooks
	pip install -e ".[dev]"
	pre-commit install

lint:  ## static checks
	ruff check src tests

fmt:  ## auto-format and fix
	ruff format src tests
	ruff check --fix src tests

test:  ## run the unit tests (no AWS needed; S3 is mocked)
	pytest

up:  ## start LocalStack
	docker compose up -d

down:  ## stop LocalStack
	docker compose down

replay:  ## stream synthetic telemetry into LocalStack S3
	AWS_ENDPOINT_URL=http://localhost:4566 AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
	python -m plantwatch.ingest.replay --source synthetic --rows 5000 --batch-size 200

demo: up replay  ## bring up LocalStack and push a stream through it

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info

.PHONY: install lint fmt test up down replay demo clean export lambda-zip tf-local tf-aws tf-destroy serve-test

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

export:  ## train and save a servable model bundle
	python -m plantwatch.pipelines.export --model pca --out artifacts/model.joblib

lambda-zip:  ## build the Lambda deployment package (model + source + deps)
	bash scripts/build_lambda.sh

tf-local:  ## deploy the stack to LocalStack (needs tflocal + Docker)
	cd infra && tflocal init && tflocal apply -auto-approve

tf-aws:  ## deploy the stack to real AWS (review the plan before approving)
	cd infra && terraform init && terraform apply

tf-destroy:  ## tear the stack down (run after a real-AWS session!)
	cd infra && terraform destroy

clean:  ## remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info build artifacts

# Convenience wrapper. Everything here also works as a plain docker/npm command.
SHELL := /bin/bash

.PHONY: help up down logs migrate revision seed test test-backend test-frontend \
        lint fmt build clean psql

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start the dev stack (db + api + web)
	docker compose up --build

down: ## Stop the stack and drop volumes
	docker compose down -v

logs: ## Tail backend logs
	docker compose logs -f backend

migrate: ## Apply database migrations
	docker compose run --rm backend alembic upgrade head

revision: ## Autogenerate a migration: make revision m="add table"
	docker compose run --rm backend alembic revision --autogenerate -m "$(m)"

seed: ## Load the demo user + sample document
	docker compose run --rm backend python -m app.cli seed

test: test-backend test-frontend ## Run the whole test suite

test-backend: ## Run backend tests against a throwaway Postgres
	docker compose run --rm -e LLM_PROVIDER=fake backend pytest -q

test-frontend: ## Run frontend unit tests
	docker compose run --rm frontend npm run test -- --run

lint: ## Lint backend and frontend
	docker compose run --rm backend sh -c "ruff check . && ruff format --check . && mypy app"
	docker compose run --rm frontend sh -c "npm run lint && npm run typecheck"

fmt: ## Auto-format the backend
	docker compose run --rm backend ruff format .

build: ## Build production images
	docker compose -f docker-compose.prod.yml build

psql: ## Open a psql shell
	docker compose exec db psql -U $${POSTGRES_USER:-rag} -d $${POSTGRES_DB:-rag}

clean: ## Remove caches and build output
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/.mypy_cache \
	       frontend/dist frontend/coverage

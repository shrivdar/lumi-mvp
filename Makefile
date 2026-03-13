.PHONY: build dev down test lint migrate seed clean

build:
	docker compose build

dev:
	docker compose up

down:
	docker compose down

test:
	cd backend && python -m pytest -v
	cd frontend && npm test -- --run

lint:
	cd backend && ruff check .
	cd frontend && npm run lint

migrate:
	cd backend && alembic upgrade head

seed:
	cd backend && python -m scripts.seed_demo

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

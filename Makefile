.PHONY: build dev down test lint migrate seed clean mcp-up mcp-down benchmark mcp-add-tool prod

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

# ── MCP ─────────────────────────────────────────

mcp-up:
	docker compose -f docker-compose.yml -f docker-compose.mcp.yml up

mcp-down:
	docker compose -f docker-compose.yml -f docker-compose.mcp.yml down

mcp-add-tool:
	@read -p "Tool name (e.g. kegg): " name; \
	mkdir -p infra/mcp/$$name && \
	cp infra/mcp/tool-template/Dockerfile infra/mcp/$$name/Dockerfile && \
	cp infra/mcp/tool-template/requirements.txt infra/mcp/$$name/requirements.txt && \
	cp infra/mcp/tool-template/server.py infra/mcp/$$name/server.py && \
	cp infra/mcp/tool-template/tool.json infra/mcp/$$name/tool.json && \
	sed -i.bak 's/example-tool/'"$$name"'/g' infra/mcp/$$name/tool.json && rm -f infra/mcp/$$name/tool.json.bak && \
	echo "Created infra/mcp/$$name — edit server.py and add service to docker-compose.mcp.yml"

# ── Benchmark ───────────────────────────────────

benchmark:
	docker compose -f docker-compose.yml -f docker-compose.benchmark.yml run --rm benchmark

# ── Production ──────────────────────────────────

prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

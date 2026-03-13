# ── Dev stage ────────────────────────────────────
FROM python:3.11-slim AS dev

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r yohas && useradd -r -g yohas -d /app -s /sbin/nologin yohas

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .
RUN chown -R yohas:yohas /app

USER yohas

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ── Prod stage ───────────────────────────────────
FROM python:3.11-slim AS prod

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r yohas && useradd -r -g yohas -d /app -s /sbin/nologin yohas

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .
RUN chown -R yohas:yohas /app

USER yohas

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

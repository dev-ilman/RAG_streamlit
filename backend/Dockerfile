# syntax=docker/dockerfile:1

# ---- base: shared runtime layer -------------------------------------------
FROM python:3.15-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# ---- deps: resolve dependencies once, cached on pyproject.toml alone ------
FROM base AS deps
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install .

# ---- dev: adds test/lint tooling, source is bind-mounted ------------------
FROM deps AS dev
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pytest pytest-asyncio pytest-cov ruff mypy
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ---- runtime: slim, non-root production image -----------------------------
FROM deps AS runtime
COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data/uploads \
 && chown -R appuser:appuser /app
USER appuser
ENV PYTHONPATH=/app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# syntax=docker/dockerfile:1.7
# The routing API on CPU. Weights are not baked in: the pinned checkpoint downloads on first start into the
# Hugging Face cache (a volume in docker-compose.yml), so the image stays small and restarts are fast.

ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv
# git: the kev dependency is pinned to a commit of its repository
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# dependencies first, so code changes do not reinstall torch
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra kev --extra service
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra kev --extra service

FROM python:${PYTHON_VERSION}-slim-bookworm
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
COPY --from=builder --chown=app:app /app /app
WORKDIR /app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    NEPKEV_HOST=0.0.0.0 \
    NEPKEV_PORT=8000
USER app
EXPOSE 8000
# Ready = model loaded and database reachable. The first start downloads ~2 GB of weights, hence the long start period.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]
CMD ["python", "-m", "nepkev.service"]

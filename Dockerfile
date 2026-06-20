ARG BASE_IMAGE=python:3.12-slim

FROM node:20-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN set -ex; \
    apt-get update && apt-get install -y --no-install-recommends gcc g++ libxrender-dev; \
    pip install --no-cache-dir uv; \
    uv pip install --system --no-cache-dir --default-timeout=300 -e .; \
    pip uninstall -y uv 2>/dev/null; \
    apt-get purge -y gcc g++ libxrender-dev; \
    apt-get autoremove --purge -y; \
    rm -rf /var/lib/apt/lists/*; \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local -name "*.pyc" -delete 2>/dev/null || true

COPY facerecserver/ ./facerecserver/
COPY scripts/ ./scripts/
COPY --from=frontend-builder /build/dist ./frontend/dist

ENV PYTHONPATH=/app
ENV FACEREC_CONFIG=/app/facerecserver/config.yaml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD python3 -c "import urllib.request; r=urllib.request.urlopen('http://localhost:8000/api/v1/livez'); assert r.status==200" || exit 1

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

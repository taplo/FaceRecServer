ARG BASE_IMAGE=python:3.12-slim

FROM node:20-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir --default-timeout=300 -e .

COPY facerecserver/ ./facerecserver/
COPY scripts/ ./scripts/
COPY --from=frontend-builder /build/dist ./frontend/dist

# Remove build dependencies (only needed at build time)
RUN apt-get remove -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Clean up Python cache files
RUN find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
RUN find /usr/local -name "*.pyc" -delete 2>/dev/null || true

ENV PYTHONPATH=/app
ENV FACEREC_CONFIG=/app/facerecserver/config.yaml
EXPOSE 8000

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

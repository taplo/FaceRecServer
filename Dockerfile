FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY facerecserver/ ./facerecserver/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV FACEREC_CONFIG=/app/facerecserver/config.yaml
EXPOSE 8000

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

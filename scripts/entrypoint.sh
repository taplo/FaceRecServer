#!/bin/bash
set -e

DEFAULT_MODEL="swin_arcface_webface4m_tinyface"
MODEL_DIR="/app/models/$DEFAULT_MODEL"
MODEL_FILE="$MODEL_DIR/model.pt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "[entrypoint] 模型未找到，开始下载: $DEFAULT_MODEL"
    uv run python scripts/download_model.py --model "$DEFAULT_MODEL" --output-dir /app/models
    echo "[entrypoint] 模型下载完成"
else
    echo "[entrypoint] 模型已存在: $MODEL_FILE"
fi

echo "[entrypoint] 启动 FaceRecServer"
exec uv run uvicorn facerecserver.app:create_app --factory --host 0.0.0.0 --port 8000

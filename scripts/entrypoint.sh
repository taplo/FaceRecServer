#!/bin/bash
set -e

# 自动检测 CPU 指令集并设置 MKL 兼容模式
MKL_FLAGS=$(python3 -c "
import platform
if platform.system() == 'Linux':
    with open('/proc/cpuinfo') as f:
        flags = f.read().lower()
    if 'avx2' in flags:
        print('AVX2')
    elif 'avx' in flags:
        print('AVX')
    else:
        print('NONE')
else:
    print('NONE')
")
if [ "$MKL_FLAGS" = "AVX" ]; then
    export MKL_CBWR=COMPATIBLE
    export MKL_ENABLE_INSTRUCTIONS=AVX
    echo "[entrypoint] CPU 不支持 AVX2，已设置 MKL 兼容模式"
elif [ "$MKL_FLAGS" = "AVX2" ]; then
    echo "[entrypoint] CPU 支持 AVX2，已启用加速"
fi

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
exec python -m uvicorn facerecserver.app:create_app --factory --host 0.0.0.0 --port 8000

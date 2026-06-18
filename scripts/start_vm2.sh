#!/bin/bash
export MKL_CBWR=COMPATIBLE
export MKL_ENABLE_INSTRUCTIONS=AVX
export UVICORN_PORT=8001
cd /home/taplo/FaceRecServer
exec /home/taplo/FaceRecServer/.venv/bin/python -m facerecserver

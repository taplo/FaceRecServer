#!/bin/bash
set -e
ssh taplo@192.168.3.123 bash -c '
cd /home/taplo/FaceRecServer
.venv/bin/pip install -e . --no-deps
sed -i "s|exec /home/taplo/FaceRecServer/.venv/bin/python -m uvicorn facerecserver.app:create_app --factory|exec /home/taplo/FaceRecServer/.venv/bin/python -m facerecserver|" /home/taplo/FaceRecServer/start_vm2.sh
pkill -9 -f facerec
pkill -9 -f uvicorn
sleep 3
tmux kill-session -t facerec 2>/dev/null
tmux new-session -d -s facerec /home/taplo/FaceRecServer/start_vm2.sh
echo "VM2 restarted"
'

#!/bin/bash
# Start CARLA 0.9.13 on a specified RPC port.
# Usage: start_carla.sh [PORT]
set -euo pipefail
PORT="${1:-2000}"
LOG="${CARLA_LOG:-/root/autodl-tmp/carla_server_${PORT}.log}"
chmod 711 /root
export XDG_RUNTIME_DIR=/tmp/carla-runtime
rm -f "$LOG"
touch "$LOG"
chown carla:carla "$LOG"
su carla -c "export XDG_RUNTIME_DIR=/tmp/carla-runtime; nohup /root/autodl-tmp/carla_server/CARLA_0.9.13/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=${PORT} &> ${LOG} &"
echo "[start_carla] CARLA starting on port ${PORT}, log=${LOG}"
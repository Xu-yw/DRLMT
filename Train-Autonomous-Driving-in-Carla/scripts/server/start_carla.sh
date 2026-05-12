#!/bin/bash
# Start CARLA 0.9.13 on a specified RPC port.
# Usage: start_carla.sh [PORT]
set -euo pipefail
PORT="${1:-2000}"
LOG="${CARLA_LOG:-/root/autodl-tmp/carla_server_${PORT}.log}"
chmod 711 /root
export XDG_RUNTIME_DIR="/tmp/carla-runtime-${PORT}"
mkdir -p "$XDG_RUNTIME_DIR"
chown carla:carla "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
rm -f "$LOG"
touch "$LOG"
chown carla:carla "$LOG"
su carla -c "export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}; export SDL_AUDIODRIVER=dummy; nohup /root/autodl-tmp/carla_server/CARLA_0.9.13/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=${PORT} &> ${LOG} &"
echo "[start_carla] CARLA starting on port ${PORT}, log=${LOG}"

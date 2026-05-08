#!/bin/bash
# plan-06 watchdog v2
# 用法: watchdog_v2.sh <mutation_type> <seed> [reward_threshold]
# 环境变量可覆盖: PORT (默认 2000)
set -uo pipefail

MUT_TYPE="${1:-baseline}"
SEED="${2:-0}"
REWARD_THRESHOLD="${3:-0}"
PORT="${PORT:-2000}"

LOG_DIR="/root/autodl-tmp/runs/${MUT_TYPE}/seed${SEED}"
DONE_FLAG="${LOG_DIR}/training_done.flag"
EP_LOG="${LOG_DIR}/episode_log.csv"
TRAIN_LOG="${LOG_DIR}/train.log"
TRAJ_LOG="${LOG_DIR}/state_traj.csv"

mkdir -p "$LOG_DIR"

cycle=0
while true; do
    cycle=$((cycle + 1))

    if [ -f "$DONE_FLAG" ]; then
        echo "[WATCHDOG][cycle $cycle] DONE flag exists, exit 0"
        exit 0
    fi

    # 启动 CARLA
    if ! pgrep -f "carla-rpc-port=$PORT" > /dev/null; then
        echo "[WATCHDOG][cycle $cycle] starting CARLA on port $PORT"
        bash /root/autodl-tmp/start_carla.sh "$PORT" > "${LOG_DIR}/carla_${PORT}.log" 2>&1 &
        sleep 30
    fi

    cd /root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla
    echo "[WATCHDOG][cycle $cycle] starting training (MUT=$MUT_TYPE SEED=$SEED THRESH=$REWARD_THRESHOLD)"

    MUTATION_TYPE="$MUT_TYPE" \
    MUTATION_SEED="$SEED" \
    TRAINING_SEED="$SEED" \
    REWARD_THRESHOLD="$REWARD_THRESHOLD" \
    EPISODE_LOG_PATH="$EP_LOG" \
    STATE_TRAJ_LOG_PATH="$TRAJ_LOG" \
    TRAINING_DONE_FLAG="$DONE_FLAG" \
    /root/miniconda3/envs/DRLMutation/bin/python continuous_driver.py \
        --exp-name ppo \
        --town Town07 \
        --seed "$SEED" \
        --total-timesteps 10000000 \
        2>&1 | tee -a "$TRAIN_LOG"

    if [ -f "$DONE_FLAG" ]; then
        echo "[WATCHDOG][cycle $cycle] training reported done, exit 0"
        exit 0
    fi

    echo "[WATCHDOG][cycle $cycle] training crashed/exited, sleeping 30s before restart"
    pkill -f "carla-rpc-port=$PORT" || true
    sleep 30
done

#!/bin/bash
# Vectorized CARLA watchdog: one learner process, multiple CARLA RPC ports.
# Usage: watchdog_vec2.sh <run_id> <seed> [reward_threshold]
# Env overrides:
#   PORTS, TOTAL_TIMESTEPS, MAX_EPISODES, CHECK_INTERVAL, STALE_SECONDS,
#   CARLA_STARTUP_SECONDS, STEP_TIMEOUT, LOG_DIR
set -u

RUN_ID="${1:-baseline_vec2_s0}"
SEED="${2:-0}"
REWARD_THRESHOLD="${3:-0}"
PORTS="${PORTS:-2002,2004}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-200000}"
MAX_EPISODES="${MAX_EPISODES:-80}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
STALE_SECONDS="${STALE_SECONDS:-240}"
CARLA_STARTUP_SECONDS="${CARLA_STARTUP_SECONDS:-60}"
STEP_TIMEOUT="${STEP_TIMEOUT:-120}"

REPO_DIR="/root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/runs/${RUN_ID}/seed${SEED}}"
DONE_FLAG="${LOG_DIR}/training_done.flag"
EP_LOG="${EPISODE_LOG_PATH:-${LOG_DIR}/episode_log.csv}"
TRAIN_LOG="${LOG_DIR}/train.log"
WD_LOG="${LOG_DIR}/watchdog_vec2.log"
TRAJ_LOG="${STATE_TRAJ_LOG_PATH:-${LOG_DIR}/state_traj.csv}"
META_DIR="${PPO_META_CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/PPO_${RUN_ID}/Town07}"
POLICY_DIR="${REPO_DIR}/preTrained_models/ppo_${RUN_ID}_s${SEED}/Town07"

mkdir -p "$LOG_DIR"

log() {
    echo "$(date '+%F %T') [WATCHDOG_VEC2] $*" | tee -a "$WD_LOG"
}

IFS=',' read -r -a PORT_ARRAY <<< "$PORTS"

carla_alive_port() {
    local port="$1"
    pgrep -f "CarlaUE4.*carla-rpc-port=${port}" >/dev/null 2>&1
}

all_carla_alive() {
    local port
    for port in "${PORT_ARRAY[@]}"; do
        if ! carla_alive_port "$port"; then
            return 1
        fi
    done
    return 0
}

count_ep_lines() {
    if [ -f "$EP_LOG" ]; then
        wc -l < "$EP_LOG" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

checkpoint_exists() {
    find "$POLICY_DIR" -maxdepth 1 -name 'ppo_policy_*_.pth' 2>/dev/null | grep -q . && \
    find "$META_DIR" -maxdepth 1 -name 'checkpoint_ppo_*.pickle' 2>/dev/null | grep -q .
}

kill_train_pid() {
    local pid="${1:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "stopping learner pid=${pid}"
        kill "$pid" 2>/dev/null || true
        sleep 5
        if kill -0 "$pid" 2>/dev/null; then
            log "learner pid=${pid} did not exit; kill -9"
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

kill_carla_all() {
    local port
    for port in "${PORT_ARRAY[@]}"; do
        if carla_alive_port "$port"; then
            log "stopping CARLA on port ${port}"
        fi
        pkill -f "CarlaUE4.*carla-rpc-port=${port}" 2>/dev/null || true
        pkill -f "CarlaUE4.sh.*carla-rpc-port=${port}" 2>/dev/null || true
    done
    sleep 5
    for port in "${PORT_ARRAY[@]}"; do
        pkill -9 -f "CarlaUE4.*carla-rpc-port=${port}" 2>/dev/null || true
        pkill -9 -f "CarlaUE4.sh.*carla-rpc-port=${port}" 2>/dev/null || true
    done
}

start_carla_all() {
    local port
    for port in "${PORT_ARRAY[@]}"; do
        if carla_alive_port "$port"; then
            log "CARLA already alive on port ${port}"
            continue
        fi
        log "starting CARLA on port ${port}"
        CARLA_LOG="${LOG_DIR}/carla_${port}.log" bash /root/autodl-tmp/start_carla.sh "$port" >> "$WD_LOG" 2>&1 &
    done

    sleep "$CARLA_STARTUP_SECONDS"

    for port in "${PORT_ARRAY[@]}"; do
        if ! carla_alive_port "$port"; then
            log "CARLA failed to start or crashed during startup on port ${port}"
            return 1
        fi
    done
    log "all CARLA ports alive: ${PORTS}"
    return 0
}

trim_episode_log_to_checkpoint() {
    local trim_script="${REPO_DIR}/scripts/server/trim_episode_log_to_checkpoint.py"
    if [ -x "$trim_script" ] && [ -f "$EP_LOG" ]; then
        /root/miniconda3/envs/DRLMutation/bin/python "$trim_script" \
            --csv "$EP_LOG" \
            --checkpoint-dir "$META_DIR" | while read -r line; do
            log "$line"
        done
    fi
}

cycle=0
log "watchdog started: run_id=${RUN_ID} seed=${SEED} ports=${PORTS} total_timesteps=${TOTAL_TIMESTEPS} max_episodes=${MAX_EPISODES} stale=${STALE_SECONDS}s step_timeout=${STEP_TIMEOUT}s"

while true; do
    cycle=$((cycle + 1))

    if [ -f "$DONE_FLAG" ]; then
        log "DONE flag exists (${DONE_FLAG}); exit 0"
        exit 0
    fi

    kill_carla_all
    start_carla_all || { kill_carla_all; sleep 20; continue; }

    cd "$REPO_DIR" || exit 2

    load_args=()
    if checkpoint_exists; then
        load_args=(--load-checkpoint True)
        log "cycle ${cycle}: checkpoint found, resume enabled"
        trim_episode_log_to_checkpoint
    else
        log "cycle ${cycle}: no checkpoint found, start from scratch"
    fi

    before_lines=$(count_ep_lines)
    log "cycle ${cycle}: starting vec learner; ep_log_lines=${before_lines}; train_log=${TRAIN_LOG}"

    env \
        PYTHONUNBUFFERED=1 \
        VEC_RUN_ID="$RUN_ID" \
        MUTATION_TYPE="$RUN_ID" \
        MUTATION_SEED="$SEED" \
        TRAINING_SEED="$SEED" \
        CARLA_PORTS="$PORTS" \
        REWARD_THRESHOLD="$REWARD_THRESHOLD" \
        EPISODE_LOG_PATH="$EP_LOG" \
        STATE_TRAJ_LOG_PATH="$TRAJ_LOG" \
        TRAINING_DONE_FLAG="$DONE_FLAG" \
        PPO_META_CHECKPOINT_DIR="$META_DIR" \
        TENSORBOARD_RUN_DIR="${LOG_DIR}/tensorboard" \
        MAX_EPISODES="$MAX_EPISODES" \
        STEP_TIMEOUT="$STEP_TIMEOUT" \
        /root/miniconda3/envs/DRLMutation/bin/python continuous_driver_vec.py \
            --exp-name ppo \
            --town Town07 \
            --seed "$SEED" \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            --carla-ports "$PORTS" \
            --run-id "$RUN_ID" \
            --max-episodes "$MAX_EPISODES" \
            --step-timeout "$STEP_TIMEOUT" \
            "${load_args[@]}" >> "$TRAIN_LOG" 2>&1 &
    train_pid=$!
    log "cycle ${cycle}: learner pid=${train_pid}"

    last_lines="$before_lines"
    last_progress_ts=$(date +%s)

    while kill -0 "$train_pid" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"
        now=$(date +%s)

        if [ -f "$DONE_FLAG" ]; then
            log "cycle ${cycle}: DONE flag written; stopping monitor"
            wait "$train_pid" 2>/dev/null || true
            log "training completed"
            exit 0
        fi

        if ! all_carla_alive; then
            log "cycle ${cycle}: at least one CARLA port disappeared; restart full vec job"
            kill_train_pid "$train_pid"
            kill_carla_all
            break
        fi

        current_lines=$(count_ep_lines)
        if [ "$current_lines" -gt "$last_lines" ]; then
            log "cycle ${cycle}: progress ep_log_lines ${last_lines} -> ${current_lines}"
            last_lines="$current_lines"
            last_progress_ts="$now"
        else
            idle=$((now - last_progress_ts))
            if [ "$idle" -ge "$STALE_SECONDS" ]; then
                log "cycle ${cycle}: no episode progress for ${idle}s; restart full vec job"
                kill_train_pid "$train_pid"
                kill_carla_all
                break
            fi
        fi
    done

    if ! kill -0 "$train_pid" 2>/dev/null; then
        wait "$train_pid" 2>/dev/null
        status=$?
        if [ -f "$DONE_FLAG" ]; then
            log "cycle ${cycle}: learner exited with DONE flag; status=${status}; exit 0"
            exit 0
        fi
        log "cycle ${cycle}: learner exited unexpectedly; status=${status}; restart after cleanup"
        kill_carla_all
    fi

    sleep 20
done

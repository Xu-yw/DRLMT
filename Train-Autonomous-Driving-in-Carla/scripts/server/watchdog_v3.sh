#!/bin/bash
# plan-06 watchdog v3: CARLA crash + Python hang recovery
# Usage: watchdog_v3.sh <mutation_type> <seed> [reward_threshold]
# Env overrides: PORT, TOTAL_TIMESTEPS, CHECK_INTERVAL, STALE_SECONDS, CARLA_STARTUP_SECONDS
# Phase 4 (2026-05-16): per-job CARLA port + per-job meta-ckpt dir via PPO_META_CHECKPOINT_DIR
# Phase 4.2 (2026-05-18): L5 mutation probe alerting (see check_mutation_probe below)
set -u

MUT_TYPE="${1:-baseline}"
SEED="${2:-0}"
REWARD_THRESHOLD="${3:-0}"
MUTATION_INTENSITY="${MUTATION_INTENSITY:-1.0}"
PORT="${PORT:-2000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
STALE_SECONDS="${STALE_SECONDS:-300}"
CARLA_STARTUP_SECONDS="${CARLA_STARTUP_SECONDS:-45}"
REPO_DIR="/root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla"
LOG_DIR="/root/autodl-tmp/runs/${MUT_TYPE}/seed${SEED}"
META_CKPT_DIR="${REPO_DIR}/checkpoints/PPO_${MUT_TYPE}_s${SEED}/Town07"
DONE_FLAG="${LOG_DIR}/training_done.flag"
EP_LOG="${EPISODE_LOG_PATH:-${LOG_DIR}/episode_log.csv}"
TRAIN_LOG="${LOG_DIR}/train.log"
TRAJ_LOG="${STATE_TRAJ_LOG_PATH:-${LOG_DIR}/state_traj.csv}"
WD_LOG="${LOG_DIR}/watchdog.log"
CARLA_LOG="${LOG_DIR}/carla_${PORT}.log"

mkdir -p "$LOG_DIR"

log() {
    echo "$(date '+%F %T') [WATCHDOG_V3] $*" | tee -a "$WD_LOG"
}

carla_alive() {
    pgrep -f "CarlaUE4.*carla-rpc-port=${PORT}" >/dev/null 2>&1
}

count_ep_lines() {
    if [ -f "$EP_LOG" ]; then
        wc -l < "$EP_LOG" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

checkpoint_exists() {
    local model_dir="${REPO_DIR}/preTrained_models/ppo_${MUT_TYPE}_s${SEED}/Town07"
    local meta_dir="$META_CKPT_DIR"
    find "$model_dir" -maxdepth 1 -name 'ppo_policy_*_.pth' 2>/dev/null | grep -q . && \
    find "$meta_dir" -maxdepth 1 -name 'checkpoint_ppo_*.pickle' 2>/dev/null | grep -q .
}

kill_train_pid() {
    local pid="${1:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "stopping training pid=${pid}"
        kill "$pid" 2>/dev/null || true
        sleep 5
        if kill -0 "$pid" 2>/dev/null; then
            log "training pid=${pid} did not exit; kill -9"
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

kill_carla() {
    if carla_alive; then
        log "stopping CARLA on port ${PORT}"
        pkill -f "CarlaUE4.*carla-rpc-port=${PORT}" 2>/dev/null || true
        pkill -f "CarlaUE4.sh.*carla-rpc-port=${PORT}" 2>/dev/null || true
        sleep 5
        pkill -9 -f "CarlaUE4.*carla-rpc-port=${PORT}" 2>/dev/null || true
        pkill -9 -f "CarlaUE4.sh.*carla-rpc-port=${PORT}" 2>/dev/null || true
    fi
}

start_carla() {
    if carla_alive; then
        log "CARLA already alive on port ${PORT}"
        return 0
    fi
    log "starting CARLA on port ${PORT}"
    bash /root/autodl-tmp/start_carla.sh "$PORT" > "$CARLA_LOG" 2>&1 &
    sleep "$CARLA_STARTUP_SECONDS"
    if carla_alive; then
        log "CARLA is alive"
        return 0
    fi
    log "CARLA failed to start; see ${CARLA_LOG}"
    return 1
}

# L5 mutation probe: parse [MUT-PROBE] lines from train.log and alert (do NOT
# kill training) on:
#   perturbed=False                       -> wired-but-silent (intensity bug class)
#   sanity_nonzero=0.0[0-4]<digit>        -> wired-but-destroyed (StFuzS blinding class)
# First probe lines are forwarded into watchdog.log; anomalies set a
# MUTATION_PROBE_ALERT.flag in $LOG_DIR. Operator monitors that flag.
check_mutation_probe() {
    [ ! -f "$TRAIN_LOG" ] && return 0
    local probe_lines
    probe_lines=$(grep "\[MUT-PROBE\]" "$TRAIN_LOG" 2>/dev/null | head -10)
    [ -z "$probe_lines" ] && return 0
    local marker="${LOG_DIR}/.probe_seen.flag"
    if [ ! -f "$marker" ]; then
        touch "$marker"
        log "mutation probe (first appearance):"
        printf '%s\n' "$probe_lines" | while read -r pline; do log "  $pline"; done
    fi
    local alert_file="${LOG_DIR}/MUTATION_PROBE_ALERT.flag"
    [ -f "$alert_file" ] && return 0  # already alerted; idempotent
    local anomaly
    anomaly=$(printf '%s\n' "$probe_lines" | grep -E "perturbed=False|sanity_nonzero=0\.0[0-4][0-9]*([^0-9]|$)" | head -5)
    if [ -n "$anomaly" ]; then
        printf '%s\n' "$anomaly" > "$alert_file"
        log "ALERT: mutation probe anomaly detected -> $alert_file"
        printf '%s\n' "$anomaly" | while read -r aline; do log "  ANOMALY: $aline"; done
    fi
    return 0
}

cycle=0
log "watchdog started: mut=${MUT_TYPE} seed=${SEED} port=${PORT} threshold=${REWARD_THRESHOLD} intensity=${MUTATION_INTENSITY} total_timesteps=${TOTAL_TIMESTEPS} stale=${STALE_SECONDS}s meta_ckpt_dir=${META_CKPT_DIR}"

while true; do
    cycle=$((cycle + 1))

    if [ -f "$DONE_FLAG" ]; then
        log "DONE flag exists (${DONE_FLAG}); exit 0"
        exit 0
    fi

    start_carla || { kill_carla; sleep 20; continue; }

    cd "$REPO_DIR" || exit 2

    load_args=()
    if checkpoint_exists; then
        load_args=(--load-checkpoint True)
        log "cycle ${cycle}: checkpoint found, resume enabled"
        trim_script="${REPO_DIR}/scripts/server/trim_episode_log_to_checkpoint.py"
        if [ -x "$trim_script" ] && [ -f "$EP_LOG" ]; then
            /root/miniconda3/envs/DRLMutation/bin/python "$trim_script" \
                --csv "$EP_LOG" \
                --checkpoint-dir "$META_CKPT_DIR" | while read -r line; do
                log "cycle ${cycle}: $line"
            done
        fi
    else
        log "cycle ${cycle}: no checkpoint found, start from scratch"
    fi

    before_lines=$(count_ep_lines)
    log "cycle ${cycle}: starting training; ep_log_lines=${before_lines}; train_log=${TRAIN_LOG}"

    env \
        PYTHONUNBUFFERED=1 \
        MUTATION_TYPE="$MUT_TYPE" \
        MUTATION_SEED="$SEED" \
        MUTATION_INTENSITY="$MUTATION_INTENSITY" \
        TRAINING_SEED="$SEED" \
        REWARD_THRESHOLD="$REWARD_THRESHOLD" \
        EPISODE_LOG_PATH="$EP_LOG" \
        STATE_TRAJ_LOG_PATH="$TRAJ_LOG" \
        TRAINING_DONE_FLAG="$DONE_FLAG" \
        PPO_META_CHECKPOINT_DIR="$META_CKPT_DIR" \
        /root/miniconda3/envs/DRLMutation/bin/python -u continuous_driver.py \
            --exp-name ppo \
            --town Town07 \
            --seed "$SEED" \
            --port "$PORT" \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            "${load_args[@]}" >> "$TRAIN_LOG" 2>&1 &
    train_pid=$!
    log "cycle ${cycle}: training pid=${train_pid}"

    last_lines="$before_lines"
    last_progress_ts=$(date +%s)

    while kill -0 "$train_pid" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"
        now=$(date +%s)

        check_mutation_probe

        if [ -f "$DONE_FLAG" ]; then
            log "cycle ${cycle}: DONE flag written; stopping monitor"
            wait "$train_pid" 2>/dev/null || true
            log "training completed"
            exit 0
        fi

        if ! carla_alive; then
            log "cycle ${cycle}: CARLA disappeared; killing training and restarting"
            kill_train_pid "$train_pid"
            kill_carla
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
                log "cycle ${cycle}: no episode progress for ${idle}s; assume Python/CARLA hang; restart"
                kill_train_pid "$train_pid"
                kill_carla
                break
            fi
        fi
    done

    if ! kill -0 "$train_pid" 2>/dev/null; then
        wait "$train_pid" 2>/dev/null
        status=$?
        if [ -f "$DONE_FLAG" ]; then
            log "cycle ${cycle}: training exited with DONE flag; status=${status}; exit 0"
            exit 0
        fi
        log "cycle ${cycle}: training exited unexpectedly; status=${status}; restart after cleanup"
        kill_carla
    fi

    sleep 20
done

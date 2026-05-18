#!/bin/bash
# Phase 4: 12 mutant 串行训练 wrapper（单 3090 + 15 vCPU 适配，max-parallel=1）
#
# Usage:
#   nohup bash /root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla/scripts/server/run_mutants_serial.sh \
#       > /root/autodl-tmp/runs/mutants/run_serial.log 2>&1 &
#
# Env overrides:
#   SEED=0
#   REWARD_THRESHOLD=1700  (来自 Phase 1 baseline)
#   TOTAL_TIMESTEPS=1000000 (1M timesteps 预算)
#   PORT=2000              (单 CARLA 端口)
#   MUTANT_QUEUE="op1 op2 ..." (空格分隔，覆盖默认 12 个)
#
# 行为:
#   - 顺序遍历 MUTANT_QUEUE
#   - 每个 mutant 跳过 done_flag 已存在的（断点续训友好）
#   - 调 /root/autodl-tmp/watchdog.sh 跑当前 mutant
#   - watchdog 退出后写一行到 queue_status.csv，再启下一个
#   - 全部跑完后 wrapper 退出
set -u

SEED="${SEED:-0}"
REWARD_THRESHOLD="${REWARD_THRESHOLD:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
PORT="${PORT:-2000}"

DEFAULT_QUEUE="StFuzS StDistP StDisoP StRepP AcRepR AcFuzS AcDisoR ReRepP ReDistP ReDisoP PVDistR ESRemS"
QUEUE="${MUTANT_QUEUE:-$DEFAULT_QUEUE}"

RUNS_ROOT="/root/autodl-tmp/runs"
STATUS_CSV="${RUNS_ROOT}/mutants/queue_status.csv"
WRAPPER_LOG="${RUNS_ROOT}/mutants/run_serial.log"

mkdir -p "$(dirname "$STATUS_CSV")"
if [ ! -f "$STATUS_CSV" ]; then
    echo "mutation_type,seed,port,started_at,ended_at,exit_code,stop_reason" > "$STATUS_CSV"
fi

log() {
    echo "$(date '+%F %T') [RUN-SERIAL] $*"
}

log "wrapper started: pid=$$ queue=[$QUEUE]"
log "config: seed=$SEED port=$PORT threshold=$REWARD_THRESHOLD budget=$TOTAL_TIMESTEPS"
log "status_csv=$STATUS_CSV"

total=$(echo "$QUEUE" | wc -w)
idx=0
for MUT in $QUEUE; do
    idx=$((idx + 1))
    DONE_FLAG="${RUNS_ROOT}/${MUT}/seed${SEED}/training_done.flag"

    if [ -f "$DONE_FLAG" ]; then
        log "[$idx/$total] $MUT: done flag exists (${DONE_FLAG}), skip"
        echo "${MUT},${SEED},${PORT},,$(date '+%F %T'),0,skipped_already_done" >> "$STATUS_CSV"
        continue
    fi

    STARTED=$(date '+%F %T')
    log "[$idx/$total] $MUT: START (port=$PORT) at $STARTED"

    PORT="$PORT" \
        TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
        bash /root/autodl-tmp/watchdog.sh "$MUT" "$SEED" "$REWARD_THRESHOLD"
    EXIT=$?
    ENDED=$(date '+%F %T')

    STOP_REASON="exit_${EXIT}"
    if [ -f "$DONE_FLAG" ]; then
        STOP_REASON="done_flag_written"
    fi

    echo "${MUT},${SEED},${PORT},${STARTED},${ENDED},${EXIT},${STOP_REASON}" >> "$STATUS_CSV"
    log "[$idx/$total] $MUT: END exit=$EXIT reason=$STOP_REASON at $ENDED"
done

log "all $total mutants processed; wrapper exit 0"

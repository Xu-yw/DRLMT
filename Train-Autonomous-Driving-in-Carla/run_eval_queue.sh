#!/bin/bash
# Eval queue runner: iterate candidates × N episodes, each writes its own CSV.
#
# - Auto-resume: skips a candidate if its CSV already exists (delete to redo).
# - Uses an INDEPENDENT CARLA on PORT (default 2002), NOT the training port 2000.
# - Caller is responsible for ensuring a CARLA server is up on $PORT.
#
# Usage:
#   bash run_eval_queue.sh                      # default 30 episodes, port 2002
#   EPISODES=5 PORT=2002 bash run_eval_queue.sh # smoke test
#   CANDIDATES_DIR=... OUT_DIR=... bash run_eval_queue.sh

set -uo pipefail

CANDIDATES_DIR="${CANDIDATES_DIR:-/root/autodl-tmp/eval/candidates}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/eval/results}"
EPISODES="${EPISODES:-30}"
PORT="${PORT:-2002}"
TOWN="${TOWN:-Town07}"
DRLMT_DIR="${DRLMT_DIR:-/root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla}"

mkdir -p "$OUT_DIR"

eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda activate DRLMutation

cd "$DRLMT_DIR"

echo "[QUEUE] candidates=$CANDIDATES_DIR results=$OUT_DIR episodes=$EPISODES port=$PORT town=$TOWN"

declare -a ORDER=(baseline_original mutant_StRepP mutant_ReDistP mutant_AcRepP mutant_EMRemP mutant_RCDistP)

for cand in "${ORDER[@]}"; do
    dir="$CANDIDATES_DIR/$cand"
    if [ ! -d "$dir" ]; then
        echo "[SKIP] $cand: directory missing"
        continue
    fi
    weight=$(ls "$dir"/ppo_policy_*_.pth 2>/dev/null | sort -V | tail -1)
    if [ -z "$weight" ]; then
        echo "[SKIP] $cand: no .pth in $dir"
        continue
    fi
    out_csv="$OUT_DIR/${cand}.csv"
    log_file="$OUT_DIR/${cand}.log"
    if [ -f "$out_csv" ] && [ -s "$out_csv" ]; then
        echo "[SKIP] $cand: $out_csv already exists (delete to redo)"
        continue
    fi
    echo "[RUN ] $cand <- $weight"
    python evaluate_only.py \
        --weight-path "$weight" \
        --episodes "$EPISODES" \
        --town "$TOWN" \
        --port "$PORT" \
        --output-csv "$out_csv" \
        --label "$cand" \
        2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] $cand exited with $rc; continuing to next"
    fi
done

echo "[QUEUE] done. CSVs in $OUT_DIR"
ls -la "$OUT_DIR"

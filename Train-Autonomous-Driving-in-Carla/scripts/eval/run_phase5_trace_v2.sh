#!/usr/bin/env bash
# Phase 5 trace v2 正式评估入口：边评估边记录 candidate_fail 轨迹。
set -euo pipefail

REPO="/root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla"
RESULT_ROOT="/root/autodl-tmp/eval/results/phase5_main_trace_v2"
TRACE_ROOT="/root/autodl-tmp/eval/results/phase5_main_trace_v2_traces"

/root/miniconda3/envs/DRLMutation/bin/python "${REPO}/scripts/eval/phase5_eval_runner.py" \
  --manifest /root/autodl-tmp/eval/candidates/phase5_main/manifest.csv \
  --test-cases /root/autodl-tmp/eval/test_cases_v1.json \
  --output-root "${RESULT_ROOT}" \
  --status-csv "${RESULT_ROOT}/phase5_queue_status.csv" \
  --trace-root "${TRACE_ROOT}" \
  --trace-mode failures \
  --trace-min-progress 0.98 \
  --suites rainy,foggy \
  --candidates baseline_v2,mutant_StDisoP,mutant_AcRepR,mutant_ReRepP,mutant_PVDistR,mutant_StRepP,mutant_StDistP,mutant_StFuzS,mutant_AcFuzS,mutant_ReDistP \
  --limit 100 \
  --port 2002 \
  --max-attempts 20 \
  --watchdog-interval 30 \
  --idle-timeout 1800 \
  --carla-missing-timeout 120 \
  --restart-carla-on-fail \
  --observer \
  --observer-web-port 8090

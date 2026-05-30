#!/usr/bin/env bash
# Phase 5 正式评估入口：baseline + 9 个 pass mutant，rainy/foggy 各 100 hard cases。
set -euo pipefail

/root/miniconda3/envs/DRLMutation/bin/python /root/autodl-tmp/eval/scripts/phase5_eval_runner.py \
  --manifest /root/autodl-tmp/eval/candidates/phase5_main/manifest.csv \
  --test-cases /root/autodl-tmp/eval/test_cases_v1.json \
  --output-root /root/autodl-tmp/eval/results/phase5_main \
  --status-csv /root/autodl-tmp/eval/results/phase5_main/phase5_queue_status.csv \
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

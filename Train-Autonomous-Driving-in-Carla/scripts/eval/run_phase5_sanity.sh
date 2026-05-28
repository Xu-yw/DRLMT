#!/usr/bin/env bash
# Phase 5 短 sanity：验证 baseline sunny、一个 admitted mutant sunny、baseline rainy。
set -euo pipefail

OUT=/root/autodl-tmp/runs/phase5_runner_sanity_20260528
mkdir -p "$OUT"

/root/miniconda3/envs/DRLMutation/bin/python /root/autodl-tmp/eval/scripts/phase5_eval_runner.py \
  --manifest /root/autodl-tmp/eval/candidates/phase5_main/manifest.csv \
  --test-cases /root/autodl-tmp/eval/test_cases_v1.json \
  --output-root "$OUT" \
  --status-csv "$OUT/phase5_sanity_status.csv" \
  --tasks baseline_v2:sunny,mutant_StRepP:sunny,baseline_v2:rainy \
  --limit 2 \
  --port 2002 \
  --max-attempts 2 \
  --observer \
  --observer-web-port 8090

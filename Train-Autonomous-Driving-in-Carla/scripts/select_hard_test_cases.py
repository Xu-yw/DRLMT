#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 baseline 在 rainy/foggy 上的筛选结果，覆盖生成正式测试用例。

脚本只读取 baseline 结果，不读取 mutant 结果，避免测试用例选择泄露变异体表现。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select hard rainy/foggy test cases using baseline only")
    parser.add_argument("--input-cases", default="/root/autodl-tmp/eval/test_cases_v1.json")
    parser.add_argument("--baseline-root", default="/root/autodl-tmp/eval/results/phase5_screening/baseline_v2")
    parser.add_argument("--output", default="/root/autodl-tmp/eval/test_cases_v1.json")
    parser.add_argument("--suites", default="rainy,foggy")
    parser.add_argument("--n-select", type=int, default=100)
    parser.add_argument("--min-progress", type=float, default=0.98)
    parser.add_argument("--max-per-spawn", type=int, default=3)
    return parser.parse_args()


def load_cases(path: str) -> Dict[int, Dict[str, object]]:
    with open(path) as f:
        cases = json.load(f)
    return {int(case["case_id"]): case for case in cases}


def load_suite_rows(path: str) -> Dict[int, Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(row["case_id"]): row for row in rows}


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def baseline_valid(rows: Iterable[Dict[str, str]], min_progress: float) -> bool:
    for row in rows:
        if row.get("done_reason") != "route_completed":
            return False
        if as_float(row, "progress_ratio") < min_progress:
            return False
    return True


def build_candidates(cases: Dict[int, Dict[str, object]], suite_rows: Dict[str, Dict[int, Dict[str, str]]],
                     suites: List[str], min_progress: float) -> List[Dict[str, object]]:
    candidates = []
    for case_id, case in cases.items():
        if any(case_id not in suite_rows[suite] for suite in suites):
            continue
        rows = [suite_rows[suite][case_id] for suite in suites]
        if not baseline_valid(rows, min_progress):
            continue
        avg_steps = sum(as_float(row, "total_steps") for row in rows) / len(rows)
        avg_reward = sum(as_float(row, "total_reward") for row in rows) / len(rows)
        candidates.append({
            "source_case": case,
            "source_case_id": case_id,
            "spawn_idx": int(case["spawn_idx"]),
            "abs_heading": abs(float(case["heading_offset_deg"])),
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
        })
    return candidates


def score_candidates(candidates: List[Dict[str, object]]) -> None:
    if not candidates:
        return
    headings = [float(item["abs_heading"]) for item in candidates]
    steps = [float(item["avg_steps"]) for item in candidates]
    rewards = [float(item["avg_reward"]) for item in candidates]
    h_min, h_max = min(headings), max(headings)
    s_min, s_max = min(steps), max(steps)
    r_min, r_max = min(rewards), max(rewards)
    for item in candidates:
        heading_score = normalize(float(item["abs_heading"]), h_min, h_max)
        steps_pressure = normalize(float(item["avg_steps"]), s_min, s_max)
        reward_pressure = 1.0 - normalize(float(item["avg_reward"]), r_min, r_max)
        item["hard_score"] = 0.50 * heading_score + 0.25 * steps_pressure + 0.25 * reward_pressure


def select_cases(candidates: List[Dict[str, object]], n_select: int, max_per_spawn: int) -> List[Dict[str, object]]:
    score_candidates(candidates)
    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("hard_score", 0.0)), float(item["abs_heading"]), float(item["avg_steps"])),
        reverse=True,
    )
    selected = []
    spawn_counts: Dict[int, int] = {}
    for item in ranked:
        spawn_idx = int(item["spawn_idx"])
        if spawn_counts.get(spawn_idx, 0) >= max_per_spawn:
            continue
        selected.append(item)
        spawn_counts[spawn_idx] = spawn_counts.get(spawn_idx, 0) + 1
        if len(selected) >= n_select:
            break
    return selected


def remap_cases(selected: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for new_id, item in enumerate(selected):
        case = dict(item["source_case"])
        case["case_id"] = new_id
        output.append(case)
    return output


def summarize(output_cases: List[Dict[str, object]], selected: List[Dict[str, object]]) -> Dict[str, object]:
    spawn_counts: Dict[int, int] = {}
    for case in output_cases:
        spawn_idx = int(case["spawn_idx"])
        spawn_counts[spawn_idx] = spawn_counts.get(spawn_idx, 0) + 1
    headings = [abs(float(case["heading_offset_deg"])) for case in output_cases]
    scores = [float(item.get("hard_score", 0.0)) for item in selected]
    return {
        "n_cases": len(output_cases),
        "unique_spawn_idx": len(spawn_counts),
        "max_per_spawn": max(spawn_counts.values()) if spawn_counts else 0,
        "min_abs_heading": min(headings) if headings else None,
        "max_abs_heading": max(headings) if headings else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
    }


def main() -> int:
    args = parse_args()
    suites = [suite for suite in args.suites.split(",") if suite]
    cases = load_cases(args.input_cases)
    suite_rows = {}
    for suite in suites:
        suite_rows[suite] = load_suite_rows(os.path.join(args.baseline_root, suite + ".csv"))
    candidates = build_candidates(cases, suite_rows, suites, args.min_progress)
    selected = select_cases(candidates, args.n_select, args.max_per_spawn)
    if len(selected) < args.n_select:
        raise RuntimeError(
            "only selected {} cases; need {}. Generate more candidates or relax baseline filters.".format(
                len(selected), args.n_select)
        )
    output_cases = remap_cases(selected)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_cases, f, indent=2)
    print("[OK] selected {} hard cases to {}".format(len(output_cases), args.output))
    print("[SUMMARY] " + json.dumps(summarize(output_cases, selected), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

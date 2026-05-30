"""
生成评估测试用例。

standard 用于普通随机用例；hard-weather 用于雨天/雾天压力测试候选用例。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def generate_cases(n_cases=500, n_spawn_points=120, seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        cases.append({
            "case_id": i,
            "spawn_idx": int(rng.integers(0, n_spawn_points)),
            "heading_offset_deg": float(rng.uniform(-15.0, 15.0)),
        })
    return cases


def _bucket_counts(n_cases):
    high = int(round(n_cases * 0.60))
    extreme = int(round(n_cases * 0.25))
    anchor = n_cases - high - extreme
    return [(high, 18.0, 28.0), (extreme, 28.0, 35.0), (anchor, 8.0, 18.0)]


def _balanced_signs(n_cases, rng):
    signs = [1] * (n_cases // 2) + [-1] * (n_cases - n_cases // 2)
    rng.shuffle(signs)
    return signs


def _spawn_sequence(n_cases, n_spawn_points, max_per_spawn, rng):
    if n_cases > n_spawn_points * max_per_spawn:
        raise ValueError("n_cases exceeds n_spawn_points * max_per_spawn")
    counts = {idx: 0 for idx in range(n_spawn_points)}
    sequence = []
    while len(sequence) < n_cases:
        candidates = [idx for idx in range(n_spawn_points) if counts[idx] < max_per_spawn]
        rng.shuffle(candidates)
        for idx in candidates:
            if len(sequence) >= n_cases:
                break
            sequence.append(idx)
            counts[idx] += 1
    return sequence


def generate_hard_weather_cases(n_cases=160, n_spawn_points=120, seed=20260530, max_per_spawn=3):
    """生成雨天/雾天候选用例，后续只用 baseline 结果筛成正式 100 cases。"""
    rng = np.random.default_rng(seed)
    headings = []
    for count, lo, hi in _bucket_counts(n_cases):
        headings.extend(float(rng.uniform(lo, hi)) for _ in range(count))
    rng.shuffle(headings)
    signs = _balanced_signs(n_cases, rng)
    spawns = _spawn_sequence(n_cases, n_spawn_points, max_per_spawn, rng)
    cases = []
    for i in range(n_cases):
        cases.append({
            "case_id": i,
            "spawn_idx": int(spawns[i]),
            "heading_offset_deg": float(signs[i] * headings[i]),
        })
    return cases


def summarize_cases(cases):
    abs_headings = [abs(float(c["heading_offset_deg"])) for c in cases]
    spawn_counts = {}
    for c in cases:
        spawn_counts[c["spawn_idx"]] = spawn_counts.get(c["spawn_idx"], 0) + 1
    positives = sum(1 for c in cases if float(c["heading_offset_deg"]) > 0)
    return {
        "n_cases": len(cases),
        "unique_spawn_idx": len(spawn_counts),
        "max_per_spawn": max(spawn_counts.values()) if spawn_counts else 0,
        "positive_heading": positives,
        "negative_heading": len(cases) - positives,
        "min_abs_heading": min(abs_headings) if abs_headings else None,
        "max_abs_heading": max(abs_headings) if abs_headings else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/root/autodl-tmp/eval/test_cases_v1.json",
                        help="Output JSON path (will mkdir parent)")
    parser.add_argument("--n-cases", type=int, default=500)
    parser.add_argument("--n-spawn-points", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-spawn-points", action="store_true",
                        help="Connect to CARLA and use len(world.get_map().get_spawn_points())")
    parser.add_argument("--profile", choices=["standard", "hard-weather"], default="standard")
    parser.add_argument("--max-per-spawn", type=int, default=3,
                        help="Maximum number of generated cases per spawn_idx")
    parser.add_argument("--town", default="Town07")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    n_spawn_points = args.n_spawn_points
    if args.detect_spawn_points:
        from simulation import settings as sim_settings
        sim_settings.PORT = args.port
        from simulation.connection import ClientConnection
        client, world = ClientConnection(args.town).setup()
        n_spawn_points = len(world.get_map().get_spawn_points())
        print(f"[INFO] detected {n_spawn_points} spawn points in {args.town}")
    if args.profile == "hard-weather":
        cases = generate_hard_weather_cases(args.n_cases, n_spawn_points, args.seed, args.max_per_spawn)
    else:
        cases = generate_cases(args.n_cases, n_spawn_points, args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2)
    summary = summarize_cases(cases)
    print(f"[OK] generated {len(cases)} {args.profile} cases to {args.output} (n_spawn_points={n_spawn_points}, seed={args.seed})")
    print("[SUMMARY] " + json.dumps(summary, sort_keys=True))
    print("First 3:")
    for c in cases[:3]:
        print(c)


if __name__ == "__main__":
    main()

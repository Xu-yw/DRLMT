#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 kill 显著性重算脚本（Phase 6-B）。

背景：首版 MS 用 Kill = (TFR_mutant > TFR_baseline) 的严格不等式，baseline TFR 恒为 0，
导致 mutant 只要 100 个 case 里错 1 个就算被杀，rainy/foggy 的 MS 全是 1.0，没有区分度。

本脚本对已有 Phase 5 per-case 结果做纯后处理（零重跑），输出三种 kill 口径并列：
  - strict ：TFR_mutant > TFR_baseline（现状口径，作对照）
  - fisher ：单边 Fisher exact test，p < alpha（用超几何分布实现，不依赖 scipy）
  - delta  ：TFR_mutant - TFR_baseline > delta_threshold（固定差值阈值，默认 0.05）

并做子集分析：从 100 个 case 里随机抽 size 个构成子套件，重算每口径下的 MS，
重复 reps 次取 mean/std/min/max，用来展示 MS 是“测试套件规模/构成”的属性，
论证 hard suite 在不同规模下的杀伤评估稳定性。

candidate_fail 口径与 compute_candidate_fail_tfr.py 完全一致，保证可比。
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from functools import lru_cache
from math import factorial
from pathlib import Path
from typing import Dict, List, Tuple


@lru_cache(maxsize=None)
def comb(n_: int, k_: int) -> int:
    """组合数 C(n, k)。DRLMutation 环境是 Python 3.7，没有 math.comb，用 factorial 实现。"""
    if k_ < 0 or k_ > n_:
        return 0
    return factorial(n_) // (factorial(k_) * factorial(n_ - k_))

# 与首版完全一致的严重失败 done_reason 集合
SEVERE_REASONS = {"collision", "lane_deviation", "low_speed_timeout", "over_speed", "step_failure", "carla_crash"}
SUITES = ("rainy", "foggy")


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_fail(row: Dict[str, str], min_progress: float) -> bool:
    """与 compute_candidate_fail_tfr.py 同口径：严重 done_reason 或 progress < 阈值。"""
    reason = (row.get("done_reason") or "").strip()
    progress = to_float(row.get("progress_ratio", ""), default=-1.0)
    return reason in SEVERE_REASONS or progress < min_progress


def fail_vector(rows: List[Dict[str, str]], min_progress: float, target_rows: int) -> List[int]:
    """按 case_id 升序返回长度 target_rows 的 0/1 失败向量，保证 baseline 与 mutant 对齐。"""
    by_case: Dict[int, int] = {}
    for row in rows:
        cid = int(to_float(row.get("case_id", ""), default=-1))
        if cid < 0:
            raise RuntimeError(f"非法 case_id: {row.get('case_id')!r}")
        by_case[cid] = 1 if candidate_fail(row, min_progress) else 0
    if len(by_case) != target_rows:
        raise RuntimeError(f"期望 {target_rows} 个唯一 case_id，实际 {len(by_case)}")
    return [by_case[c] for c in sorted(by_case)]


# ---------- Fisher exact 单边检验（超几何分布实现） ----------

def hypergeom_pmf(x: int, N: int, K: int, n: int) -> float:
    """P(X = x)，X ~ 超几何(N 总数, K 成功总数, n 抽样数)。越界返回 0。"""
    if x < 0 or x > K or (n - x) < 0 or (n - x) > (N - K):
        return 0.0
    return comb(K, x) * comb(N - K, n - x) / comb(N, n)


def fisher_one_sided_greater(base_fail: int, base_n: int, mut_fail: int, mut_n: int) -> float:
    """
    单边 Fisher exact，备择假设 H1: mutant 失败率 > baseline 失败率。
    固定 2x2 边际后，baseline 行失败数 X ~ 超几何(N=base_n+mut_n, K=总失败, n=base_n)。
    H1 成立时 baseline 失败应偏少，故取左尾 p = P(X <= base_fail)。
    """
    N = base_n + mut_n
    K = base_fail + mut_fail
    n = base_n
    if K == 0:
        return 1.0
    p = sum(hypergeom_pmf(x, N, K, n) for x in range(0, base_fail + 1))
    return min(1.0, p)


# ---------- 三口径 kill 判定 ----------

def kill_decisions(base_fail: int, mut_fail: int, n: int, alpha: float, delta_threshold: float
                   ) -> Tuple[int, int, int, float]:
    base_tfr = base_fail / n
    mut_tfr = mut_fail / n
    p = fisher_one_sided_greater(base_fail, n, mut_fail, n)
    kill_strict = 1 if mut_tfr > base_tfr else 0
    kill_fisher = 1 if p < alpha else 0
    kill_delta = 1 if (mut_tfr - base_tfr) > delta_threshold else 0
    return kill_strict, kill_fisher, kill_delta, p


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="重算 Phase 6 kill 显著性与多口径 MS（零重跑）。")
    parser.add_argument("--results-root", required=True, help="phase5_main 根目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default="baseline_v2")
    parser.add_argument("--mutants", default="", help="逗号分隔；空表示自动发现 mutant_* 目录")
    parser.add_argument("--target-rows", type=int, default=100)
    parser.add_argument("--min-progress", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--subset-sizes", default="10,20,30,50,100")
    parser.add_argument("--subset-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    n = args.target_rows

    if args.mutants.strip():
        mutants = [m.strip() for m in args.mutants.split(",") if m.strip()]
    else:
        mutants = sorted(p.name for p in results_root.iterdir()
                         if p.is_dir() and p.name.startswith("mutant_"))

    # 读取 baseline 与所有 mutant 的逐 case 失败向量
    vectors: Dict[str, Dict[str, List[int]]] = {}
    for cand in [args.baseline] + mutants:
        vectors[cand] = {}
        for suite in SUITES:
            rows = read_rows(results_root / cand / f"{suite}.csv")
            vectors[cand][suite] = fail_vector(rows, args.min_progress, n)

    # ---------- 全量（100 case）kill matrix + MS ----------
    kill_rows: List[Dict[str, object]] = []
    ms_rows: List[Dict[str, object]] = []
    for suite in SUITES:
        base_fail = sum(vectors[args.baseline][suite])
        k_strict = k_fisher = k_delta = 0
        for mut in mutants:
            mut_fail = sum(vectors[mut][suite])
            ks, kf, kd, p = kill_decisions(base_fail, mut_fail, n, args.alpha, args.delta_threshold)
            k_strict += ks
            k_fisher += kf
            k_delta += kd
            kill_rows.append({
                "suite": suite,
                "mutant": mut,
                "baseline_fail": base_fail,
                "mutant_fail": mut_fail,
                "baseline_TFR": f"{base_fail / n:.6f}",
                "mutant_TFR": f"{mut_fail / n:.6f}",
                "TFR_delta": f"{(mut_fail - base_fail) / n:.6f}",
                "fisher_p": f"{p:.6g}",
                "kill_strict": ks,
                "kill_fisher": kf,
                "kill_delta": kd,
            })
        nm = len(mutants)
        ms_rows.append({
            "suite": suite,
            "n_mutants": nm,
            "baseline_fail": base_fail,
            "MS_strict": f"{k_strict / nm:.6f}",
            "MS_fisher": f"{k_fisher / nm:.6f}",
            "MS_delta": f"{k_delta / nm:.6f}",
            "killed_strict": k_strict,
            "killed_fisher": k_fisher,
            "killed_delta": k_delta,
            "alpha": args.alpha,
            "delta_threshold": args.delta_threshold,
        })

    write_csv(output_dir / "kill_matrix_multi.csv", kill_rows, [
        "suite", "mutant", "baseline_fail", "mutant_fail", "baseline_TFR", "mutant_TFR",
        "TFR_delta", "fisher_p", "kill_strict", "kill_fisher", "kill_delta",
    ])
    write_csv(output_dir / "ms_summary_multi.csv", ms_rows, [
        "suite", "n_mutants", "baseline_fail", "MS_strict", "MS_fisher", "MS_delta",
        "killed_strict", "killed_fisher", "killed_delta", "alpha", "delta_threshold",
    ])

    # ---------- 子集规模曲线 ----------
    rng = random.Random(args.seed)
    sizes = [int(s) for s in args.subset_sizes.split(",") if s.strip()]
    subset_rows: List[Dict[str, object]] = []
    for suite in SUITES:
        for size in sizes:
            if size > n:
                continue
            ms_by_rule = {"strict": [], "fisher": [], "delta": []}
            reps = 1 if size == n else args.subset_reps
            for _ in range(reps):
                idx = list(range(n)) if size == n else rng.sample(range(n), size)
                base_fail = sum(vectors[args.baseline][suite][i] for i in idx)
                ks = kf = kd = 0
                for mut in mutants:
                    mut_fail = sum(vectors[mut][suite][i] for i in idx)
                    a, b, c, _ = kill_decisions(base_fail, mut_fail, size, args.alpha, args.delta_threshold)
                    ks += a
                    kf += b
                    kd += c
                nm = len(mutants)
                ms_by_rule["strict"].append(ks / nm)
                ms_by_rule["fisher"].append(kf / nm)
                ms_by_rule["delta"].append(kd / nm)
            for rule, vals in ms_by_rule.items():
                subset_rows.append({
                    "suite": suite,
                    "kill_rule": rule,
                    "subset_size": size,
                    "reps": reps,
                    "MS_mean": f"{statistics.mean(vals):.6f}",
                    "MS_std": f"{(statistics.pstdev(vals) if len(vals) > 1 else 0.0):.6f}",
                    "MS_min": f"{min(vals):.6f}",
                    "MS_max": f"{max(vals):.6f}",
                })

    write_csv(output_dir / "ms_subset_curve.csv", subset_rows, [
        "suite", "kill_rule", "subset_size", "reps", "MS_mean", "MS_std", "MS_min", "MS_max",
    ])

    # 控制台摘要，便于直接 ssh 看结果
    print("=== MS summary (multi-rule) ===")
    print("suite,MS_strict,MS_fisher,MS_delta")
    for r in ms_rows:
        print(f"{r['suite']},{r['MS_strict']},{r['MS_fisher']},{r['MS_delta']}")
    print(f"alpha={args.alpha} delta_threshold={args.delta_threshold} mutants={len(mutants)} seed={args.seed}")
    print(f"out={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

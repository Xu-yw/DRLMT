#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
块1 零重跑充分性论证（SC 替代证据）。

在固定路线/固定天气/固定 100 场景的设定下，用两类零重跑覆盖论证测试集充分性：
  ① 场景输入覆盖：test_cases_v1.json 的 spawn_idx 覆盖率 + heading_offset 分布
  ② 失效模式覆盖：phase5_main 各 candidate CSV 的 done_reason 分布，
     证明测试集能逼出系统的多种失效类型（而非单一模式）

配合已完成的 MS-子集规模曲线（phase6_kill_significance_20260613），三者共同替代论文 latent SC。
纯读 CSV/json，不依赖 CARLA。
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEVERE = {"collision", "lane_deviation", "low_speed_timeout", "over_speed", "step_failure", "carla_crash"}
SUITES = ("rainy", "foggy")
REASON_TYPES = ["route_completed", "max_steps_reached", "collision", "lane_deviation", "over_speed", "low_speed_timeout", "step_failure"]


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--test-cases", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-spawn-points", type=int, default=116)
    ap.add_argument("--baseline", default="baseline_v2")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    figs = os.path.join(args.output_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    # ---------- ① 场景输入覆盖 ----------
    cases = json.load(open(args.test_cases))
    spawns = [int(c["spawn_idx"]) for c in cases]
    headings = [float(c["heading_offset_deg"]) for c in cases]
    spawn_counter = Counter(spawns)
    n_cases = len(cases)
    unique_spawn = len(spawn_counter)
    abs_h = [abs(h) for h in headings]
    lines = [
        "# 块1 零重跑充分性论证（SC 替代证据）",
        "",
        "## ① 场景输入覆盖（test_cases_v1.json）",
        "",
        f"- n_cases = {n_cases}",
        f"- unique spawn = {unique_spawn} / {args.n_spawn_points}  "
        f"({100.0 * unique_spawn / args.n_spawn_points:.1f}% of Town07 spawn points)",
        f"- max cases per spawn = {max(spawn_counter.values())}",
        f"- heading_offset_deg range = [{min(headings):.2f}, {max(headings):.2f}]",
        f"- |heading_offset_deg| range = [{min(abs_h):.2f}, {max(abs_h):.2f}]",
    ]
    with open(os.path.join(args.output_dir, "input_spawn_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spawn_idx", "n_cases"])
        for s in sorted(spawn_counter):
            w.writerow([s, spawn_counter[s]])

    # ---------- ② 失效模式覆盖 ----------
    candidates = sorted(p for p in os.listdir(args.results_root)
                        if os.path.isdir(os.path.join(args.results_root, p)))
    candidates = [c for c in candidates if c == args.baseline] + [c for c in candidates if c != args.baseline]
    fm_rows = []
    all_failure_reasons = set()
    matrix = {}
    for cand in candidates:
        for suite in SUITES:
            p = os.path.join(args.results_root, cand, f"{suite}.csv")
            if not os.path.exists(p):
                continue
            rows = read_csv(p)
            dr = Counter((r.get("done_reason") or "").strip() for r in rows)
            matrix[(cand, suite)] = dr
            fails = Counter()
            for r in rows:
                reason = (r.get("done_reason") or "").strip()
                prog = to_float(r.get("progress_ratio", ""), -1.0)
                if reason in SEVERE or prog < 0.98:
                    fails[reason] += 1
                    if reason in SEVERE:
                        all_failure_reasons.add(reason)
            row = {"candidate": cand, "suite": suite, "n_cases": len(rows)}
            for t in REASON_TYPES:
                row[t] = dr.get(t, 0)
            row["candidate_fail"] = sum(fails.values())
            row["distinct_severe_modes"] = len([k for k in fails if k in SEVERE])
            fm_rows.append(row)

    with open(os.path.join(args.output_dir, "failure_mode_coverage.csv"), "w", newline="") as f:
        fn = ["candidate", "suite", "n_cases"] + REASON_TYPES + ["candidate_fail", "distinct_severe_modes"]
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in fm_rows:
            w.writerow(r)

    lines += [
        "",
        "## ② 失效模式覆盖（phase5_main per-case done_reason）",
        "",
        f"- 测试集在 9 个 mutant 上整体触发的严重失效类型 = {sorted(all_failure_reasons)}"
        f"（共 {len(all_failure_reasons)} 种）",
        "- 每个 mutant 的主导失效模式与计数见 failure_mode_coverage.csv 和热图。",
    ]

    # ---------- 图1：输入覆盖 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    used = np.zeros(args.n_spawn_points)
    for s, c in spawn_counter.items():
        if 0 <= s < args.n_spawn_points:
            used[s] = c
    ax1.bar(range(args.n_spawn_points), used, color="#1f77b4", width=1.0)
    ax1.set_xlabel("Town07 spawn point index (0..%d)" % (args.n_spawn_points - 1))
    ax1.set_ylabel("cases using this spawn")
    ax1.set_title("Input coverage: spawn (%d/%d used, %.0f%%)"
                  % (unique_spawn, args.n_spawn_points, 100.0 * unique_spawn / args.n_spawn_points))
    ax2.hist(headings, bins=20, color="#d62728", alpha=0.8)
    ax2.set_xlabel("heading_offset_deg")
    ax2.set_ylabel("count")
    ax2.set_title("Input coverage: heading offset distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "input_coverage.png"), dpi=150)
    plt.close(fig)

    # ---------- 图2：失效模式覆盖矩阵 ----------
    plot_reasons = ["route_completed", "max_steps_reached", "collision", "lane_deviation", "over_speed", "low_speed_timeout"]
    cand_list = [c for c in candidates if any((c, s) in matrix for s in SUITES)]
    M = np.zeros((len(cand_list), len(plot_reasons)))
    for i, cand in enumerate(cand_list):
        agg = Counter()
        for suite in SUITES:
            agg += matrix.get((cand, suite), Counter())
        for j, r in enumerate(plot_reasons):
            M[i, j] = agg.get(r, 0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(plot_reasons)))
    ax.set_xticklabels(plot_reasons, rotation=30, ha="right")
    ax.set_yticks(range(len(cand_list)))
    ax.set_yticklabels([c.replace("mutant_", "") for c in cand_list])
    for i in range(len(cand_list)):
        for j in range(len(plot_reasons)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Failure-mode coverage (rainy+foggy combined counts)")
    fig.colorbar(im, ax=ax, label="case count")
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "failure_mode_matrix.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "coverage_notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("out=" + args.output_dir)


if __name__ == "__main__":
    main()

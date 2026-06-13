#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 kill 显著性可视化脚本。

读取 compute_kill_significance.py 的输出，画两张图（英文标签，避免远程中文字体缺失）：
  1) ms_subset_curve.png      —— MS 随测试套件规模增长曲线（Fisher 口径，rainy/foggy 两条线带 ±std 阴影），
                                  论证 MS 是"测试套件规模/构成"的属性、100-case 规模的必要性。
  2) kill_significance.png    —— 9 个 mutant 的 Fisher 显著性（-log10(p)）分组条形图，rainy/foggy 并列，
                                  画出 p=0.05 阈值线，直观看哪些 mutant 被统计显著地杀死。

远程无显示环境，强制 Agg backend。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def short_name(mutant: str) -> str:
    return mutant.replace("mutant_", "")


def plot_subset_curve(metrics_dir: Path, out_dir: Path) -> Path:
    rows = read_csv(metrics_dir / "ms_subset_curve.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = {"rainy": "#1f77b4", "foggy": "#d62728"}
    for suite in ("rainy", "foggy"):
        pts = [r for r in rows if r["suite"] == suite and r["kill_rule"] == "fisher"]
        pts.sort(key=lambda r: int(r["subset_size"]))
        xs = [int(r["subset_size"]) for r in pts]
        ys = [float(r["MS_mean"]) for r in pts]
        sd = [float(r["MS_std"]) for r in pts]
        lo = [max(0.0, y - s) for y, s in zip(ys, sd)]
        hi = [min(1.0, y + s) for y, s in zip(ys, sd)]
        ax.plot(xs, ys, marker="o", color=colors[suite], label=f"{suite} (Fisher MS)")
        ax.fill_between(xs, lo, hi, color=colors[suite], alpha=0.15)
    ax.set_xlabel("Test suite size (number of cases)")
    ax.set_ylabel("Mutation Score (Fisher, p<0.05)")
    ax.set_title("MS grows with suite size — MS is a property of the test suite")
    ax.set_ylim(0, 1.05)
    ax.set_xticks([10, 20, 30, 50, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "ms_subset_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_kill_significance(metrics_dir: Path, out_dir: Path, alpha: float = 0.05) -> Path:
    rows = read_csv(metrics_dir / "kill_matrix_multi.csv")
    mutants = sorted({short_name(r["mutant"]) for r in rows})
    by_suite = {"rainy": {}, "foggy": {}}
    for r in rows:
        p = float(r["fisher_p"])
        # -log10(p)：越大越显著；p=0 保护
        score = -math.log10(p) if p > 0 else 50.0
        by_suite[r["suite"]][short_name(r["mutant"])] = score

    x = range(len(mutants))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], [by_suite["rainy"].get(m, 0) for m in mutants],
           width, label="rainy", color="#1f77b4")
    ax.bar([i + width / 2 for i in x], [by_suite["foggy"].get(m, 0) for m in mutants],
           width, label="foggy", color="#d62728")
    thr = -math.log10(alpha)
    ax.axhline(thr, color="black", linestyle="--", linewidth=1,
               label=f"significance threshold (p={alpha}, -log10={thr:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(mutants, rotation=30, ha="right")
    ax.set_ylabel("-log10(Fisher p)  (higher = more significant kill)")
    ax.set_title("Per-mutant kill significance (Fisher exact, vs baseline)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "kill_significance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Phase 6 kill significance figures.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir) if args.out_dir else metrics_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = plot_subset_curve(metrics_dir, out_dir)
    p2 = plot_kill_significance(metrics_dir, out_dir, args.alpha)
    print(f"saved={p1}")
    print(f"saved={p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

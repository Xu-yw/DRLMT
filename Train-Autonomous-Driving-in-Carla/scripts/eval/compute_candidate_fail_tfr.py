#!/usr/bin/env python3
"""Phase 6-A TFR ?????

??? candidate_fail ???? baseline ???? mutant ???????
?????? Phase 5 CSV ??? TFR ????? case ?????? MS?
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

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
    """????????? done_reason ??????????"""
    reason = (row.get("done_reason") or "").strip()
    progress = to_float(row.get("progress_ratio", ""), default=-1.0)
    return reason in SEVERE_REASONS or progress < min_progress


def discover_complete_candidates(results_root: Path, suites: Iterable[str], target_rows: int) -> List[str]:
    candidates: List[str] = []
    for cand_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        ok = True
        for suite in suites:
            csv_path = cand_dir / f"{suite}.csv"
            done_path = Path(str(csv_path) + ".done")
            if not csv_path.exists() or not done_path.exists():
                ok = False
                break
            if len(read_rows(csv_path)) != target_rows:
                ok = False
                break
        if ok:
            candidates.append(cand_dir.name)
    return candidates


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute candidate_fail TFR for completed Phase 5 candidates.")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-rows", type=int, default=100)
    parser.add_argument("--min-progress", type=float, default=0.98)
    parser.add_argument("--candidates", default="", help="Comma-separated candidates. Empty means auto-discover complete candidates.")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    suites = SUITES

    if args.candidates.strip():
        candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    else:
        candidates = discover_complete_candidates(results_root, suites, args.target_rows)

    summary_rows: List[Dict[str, object]] = []
    fail_rows: List[Dict[str, object]] = []
    reason_rows: List[Dict[str, object]] = []

    for candidate in candidates:
        for suite in suites:
            csv_path = results_root / candidate / f"{suite}.csv"
            rows = read_rows(csv_path)
            if len(rows) != args.target_rows:
                raise RuntimeError(f"{csv_path} has {len(rows)} rows, expected {args.target_rows}")

            failures = []
            done_reasons = Counter((row.get("done_reason") or "").strip() for row in rows)
            fail_reasons = Counter()
            progress_values = [to_float(row.get("progress_ratio", ""), default=0.0) for row in rows]
            reward_values = [to_float(row.get("total_reward", ""), default=0.0) for row in rows]

            for row in rows:
                failed = candidate_fail(row, args.min_progress)
                if failed:
                    reason = (row.get("done_reason") or "").strip()
                    fail_reasons[reason] += 1
                    failures.append(row)
                    fail_rows.append({
                        "candidate": candidate,
                        "suite": suite,
                        "case_id": row.get("case_id", ""),
                        "spawn_idx": row.get("spawn_idx", ""),
                        "heading_offset_deg": row.get("heading_offset_deg", ""),
                        "done_reason": reason,
                        "progress_ratio": row.get("progress_ratio", ""),
                        "total_reward": row.get("total_reward", ""),
                        "total_steps": row.get("total_steps", ""),
                    })

            summary_rows.append({
                "candidate": candidate,
                "suite": suite,
                "n_cases": len(rows),
                "candidate_fail_count": len(failures),
                "TFR": f"{len(failures) / args.target_rows:.6f}",
                "min_progress_ratio": f"{min(progress_values):.6f}",
                "mean_progress_ratio": f"{sum(progress_values) / len(progress_values):.6f}",
                "mean_total_reward": f"{sum(reward_values) / len(reward_values):.6f}",
                "route_completed_count": done_reasons.get("route_completed", 0),
                "max_steps_reached_count": done_reasons.get("max_steps_reached", 0),
                "collision_count": done_reasons.get("collision", 0),
                "lane_deviation_count": done_reasons.get("lane_deviation", 0),
                "low_speed_timeout_count": done_reasons.get("low_speed_timeout", 0),
                "over_speed_count": done_reasons.get("over_speed", 0),
                "candidate_fail_reasons": ";".join(f"{k}:{v}" for k, v in fail_reasons.most_common()),
            })
            for reason, count in done_reasons.most_common():
                reason_rows.append({"candidate": candidate, "suite": suite, "done_reason": reason, "count": count})

    write_csv(output_dir / "tfr_candidate_fail_summary.csv", summary_rows, [
        "candidate", "suite", "n_cases", "candidate_fail_count", "TFR",
        "min_progress_ratio", "mean_progress_ratio", "mean_total_reward",
        "route_completed_count", "max_steps_reached_count", "collision_count",
        "lane_deviation_count", "low_speed_timeout_count", "over_speed_count",
        "candidate_fail_reasons",
    ])
    write_csv(output_dir / "candidate_fail_cases.csv", fail_rows, [
        "candidate", "suite", "case_id", "spawn_idx", "heading_offset_deg",
        "done_reason", "progress_ratio", "total_reward", "total_steps",
    ])
    write_csv(output_dir / "done_reason_counts.csv", reason_rows, ["candidate", "suite", "done_reason", "count"])

    notes = output_dir / "phase6_candidate_fail_notes.md"
    notes.write_text(
        "# Phase 6-A candidate_fail TFR\n\n"
        "- MS ?????\n"
        "- ????? rainy/foggy ?? suite ???? 100 case ??? .done ??????\n"
        "- candidate_fail = done_reason ?? collision/lane_deviation/low_speed_timeout/over_speed/step_failure/carla_crash?? progress_ratio < 0.98?\n"
        "- TFR = candidate_fail_count / 100?\n",
        encoding="utf-8",
    )
    print(f"candidates={','.join(candidates)}")
    print(f"summary={output_dir / 'tfr_candidate_fail_summary.csv'}")
    print(f"fail_cases={output_dir / 'candidate_fail_cases.csv'}")
    print(f"done_reasons={output_dir / 'done_reason_counts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

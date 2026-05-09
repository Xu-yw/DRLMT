#!/usr/bin/env python3
"""Append a Plan-06 training performance snapshot to runs/perf_profile.md."""
import argparse
import csv
import datetime as dt
import os
import statistics
import subprocess
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def pgrep(pattern):
    out = run(["pgrep", "-f", pattern])
    pids = []
    for token in out.split():
        if token.isdigit():
            pids.append(token)
    return pids


def ps_table(pids):
    if not pids:
        return "(none)"
    return run(["ps", "-p", ",".join(pids), "-o", "pid,pcpu,pmem,rss,etime,cmd", "--no-headers"])


def read_episode_summary(path, window):
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    rows = []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {"exists": True, "rows": 0}
    recent = rows[-window:]
    def floats(name):
        vals = []
        for row in recent:
            try:
                vals.append(float(row[name]))
            except Exception:
                pass
        return vals
    rewards = floats("total_reward")
    steps = floats("total_steps")
    wall = floats("episode_wall_time_s")
    reasons = {}
    for row in recent:
        reason = row.get("done_reason", "") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    spawn_unique = None
    if "spawn_idx" in rows[0]:
        spawn_unique = len({row.get("spawn_idx") for row in recent if row.get("spawn_idx") not in (None, "")})
    return {
        "exists": True,
        "rows": len(rows),
        "last_episode": rows[-1].get("episode"),
        "last_timestep": rows[-1].get("timestep_at_done"),
        "reward_mean": statistics.mean(rewards) if rewards else None,
        "steps_mean": statistics.mean(steps) if steps else None,
        "wall_mean": statistics.mean(wall) if wall else None,
        "wall_p50": statistics.median(wall) if wall else None,
        "wall_p90": sorted(wall)[int(len(wall) * 0.90) - 1] if wall else None,
        "reasons": reasons,
        "spawn_unique": spawn_unique,
        "window": len(recent),
    }


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    try:
        return ("%%.%df" % digits) % value
    except Exception:
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-log", default="/root/autodl-tmp/runs/baseline/seed0/episode_log.csv")
    parser.add_argument("--output", default="/root/autodl-tmp/runs/perf_profile.md")
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    summary = read_episode_summary(args.episode_log, args.window)
    py_pids = pgrep("continuous_driver.py")
    carla_pids = pgrep("CarlaUE4")
    now = dt.datetime.now().isoformat(timespec="seconds")

    lines = []
    lines.append("## Snapshot %s" % now)
    lines.append("")
    lines.append("- episode_log: `%s`" % args.episode_log)
    if not summary.get("exists"):
        lines.append("- episode_log_status: missing")
    else:
        lines.append("- episodes: %s" % summary.get("rows", 0))
        lines.append("- last_episode: %s" % summary.get("last_episode", "n/a"))
        lines.append("- last_timestep: %s" % summary.get("last_timestep", "n/a"))
        lines.append("- recent_window: %s" % summary.get("window", 0))
        lines.append("- recent_reward_mean: %s" % fmt(summary.get("reward_mean")))
        lines.append("- recent_steps_mean: %s" % fmt(summary.get("steps_mean")))
        lines.append("- episode_wall_time_mean_s: %s" % fmt(summary.get("wall_mean")))
        lines.append("- episode_wall_time_p50_s: %s" % fmt(summary.get("wall_p50")))
        lines.append("- episode_wall_time_p90_s: %s" % fmt(summary.get("wall_p90")))
        lines.append("- recent_unique_spawn_idx: %s" % (summary.get("spawn_unique") if summary.get("spawn_unique") is not None else "n/a"))
        lines.append("- recent_done_reasons: %s" % summary.get("reasons", {}))
    lines.append("")
    lines.append("### System")
    lines.append("```text")
    lines.append(run(["bash", "-lc", "top -bn1 | head -n 8"]))
    lines.append("")
    lines.append(run(["df", "-h", "/root/autodl-tmp"]))
    lines.append("```")
    lines.append("")
    lines.append("### Python training processes")
    lines.append("```text")
    lines.append(ps_table(py_pids))
    lines.append("```")
    lines.append("")
    lines.append("### CARLA processes")
    lines.append("```text")
    lines.append(ps_table(carla_pids))
    lines.append("```")
    lines.append("")
    lines.append("### GPU")
    lines.append("```text")
    lines.append(run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]))
    lines.append("```")
    lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[OK] appended performance snapshot to %s" % out)


if __name__ == "__main__":
    main()
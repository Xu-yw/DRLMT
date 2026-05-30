#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5 评估调度器。

读取 Phase 5 manifest，按 candidate × suite 调用 evaluate_suite.py。
默认测试套件为 rainy/foggy。调度器负责：CARLA 端口检查、断点续跑、任务状态 CSV、observer 开关、
失败后的有限重试。正式 100-case 和短 sanity 共用这一入口。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


REPO = "/root/autodl-tmp/DRLMT/Train-Autonomous-Driving-in-Carla"
PYTHON = "/root/miniconda3/envs/DRLMutation/bin/python"
EVALUATE = os.path.join(REPO, "evaluate_suite.py")
DEFAULT_MANIFEST = "/root/autodl-tmp/eval/candidates/phase5_main/manifest.csv"
DEFAULT_CASES = "/root/autodl-tmp/eval/test_cases_v1.json"
DEFAULT_OUTPUT_ROOT = "/root/autodl-tmp/eval/results/phase5_main"
DEFAULT_STATUS = "/root/autodl-tmp/eval/results/phase5_main/phase5_queue_status.csv"
SUITE_WEATHER = {
    "sunny": "ClearNoon",
    "rainy": "MidRainyNoon",
    "foggy": "Custom_Foggy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 CARLA evaluation runner")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--test-cases", default=DEFAULT_CASES)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-csv", default=DEFAULT_STATUS)
    parser.add_argument("--suites", default="rainy,foggy", help="comma list: sunny,rainy,foggy")
    parser.add_argument("--candidates", default="", help="comma list from manifest; empty = all")
    parser.add_argument("--tasks", default="", help="comma list candidate:suite; overrides candidates/suites")
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases; sanity can use 1/2")
    parser.add_argument("--max-steps", type=int, default=7500)
    parser.add_argument("--port", type=int, default=2002)
    parser.add_argument("--carla-start-script", default="/root/autodl-tmp/start_carla.sh")
    parser.add_argument("--carla-wait", type=int, default=45)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--restart-carla-on-fail", action="store_true")
    parser.add_argument("--observer", action="store_true")
    parser.add_argument("--observer-web-host", default="127.0.0.1")
    parser.add_argument("--observer-web-port", type=int, default=8090)
    parser.add_argument("--watchdog-interval", type=int, default=30,
                        help="seconds between child process health checks")
    parser.add_argument("--idle-timeout", type=int, default=1800,
                        help="seconds without new CSV rows before restarting evaluation")
    parser.add_argument("--carla-missing-timeout", type=int, default=120,
                        help="seconds to wait after CARLA process disappears before restart")
    return parser.parse_args()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"candidate", "mutation_type", "staged_ckpt"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise RuntimeError("manifest missing fields: {}".format(sorted(missing)))
    return rows


def read_case_count(path: str) -> int:
    with open(path) as f:
        return len(json.load(f))


def data_rows(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    with open(path, newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def ensure_status_header(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate", "mutation_type", "suite", "weather",
            "attempt", "started_at", "ended_at", "exit_code",
            "data_rows", "target_rows", "status", "output_csv", "log_path",
        ])


def append_status(path: str, row: List[object]) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def ensure_carla(port: int, start_script: str, wait_s: int) -> None:
    probe = subprocess.run(["pgrep", "-f", "carla-rpc-port={}".format(port)])
    if probe.returncode == 0:
        return
    subprocess.check_call(["bash", start_script, str(port)])
    time.sleep(wait_s)


def restart_carla(port: int, start_script: str, wait_s: int) -> None:
    # 只杀指定 RPC 端口的 CARLA，避免影响训练端口。
    subprocess.run(["pkill", "-f", "carla-rpc-port={}".format(port)])
    time.sleep(5)
    subprocess.check_call(["bash", start_script, str(port)])
    time.sleep(wait_s)


def carla_running(port: int) -> bool:
    return subprocess.run(
        ["pgrep", "-f", "carla-rpc-port={}".format(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def terminate_process_group(proc: subprocess.Popen, log_f, reason: str) -> None:
    if proc.poll() is not None:
        return
    log_f.write("[WATCHDOG] terminating pid={} reason={}\n".format(proc.pid, reason))
    log_f.flush()
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    log_f.write("[WATCHDOG] SIGTERM timeout; sending SIGKILL pid={}\n".format(proc.pid))
    log_f.flush()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_child_with_watchdog(cmd: List[str], env: Dict[str, str], log_f, output_csv: str,
                            target_rows: int, args: argparse.Namespace) -> Tuple[int, str]:
    # 子进程单独成组，watchdog 超时时可以同时清理 evaluate 和 observer。
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    last_rows = data_rows(output_csv)
    last_progress = time.monotonic()
    carla_missing_since = None
    log_f.write("[WATCHDOG] child_pid={} start_rows={} target_rows={} idle_timeout={} carla_missing_timeout={}\n".format(
        proc.pid, last_rows, target_rows, args.idle_timeout, args.carla_missing_timeout))
    log_f.flush()

    while True:
        returncode = proc.poll()
        if returncode is not None:
            return returncode, "exit"

        now = time.monotonic()
        rows = data_rows(output_csv)
        if rows > last_rows:
            last_rows = rows
            last_progress = now
            carla_missing_since = None
            log_f.write("[WATCHDOG] progress rows={}/{} at={}\n".format(rows, target_rows, now_str()))
            log_f.flush()

        if not carla_running(args.port):
            if carla_missing_since is None:
                carla_missing_since = now
                log_f.write("[WATCHDOG] CARLA missing at {}; waiting {}s before restart\n".format(
                    now_str(), args.carla_missing_timeout))
                log_f.flush()
            elif now - carla_missing_since >= args.carla_missing_timeout:
                terminate_process_group(proc, log_f, "carla_missing_timeout")
                return -124, "carla_missing_timeout"
        else:
            carla_missing_since = None

        if now - last_progress >= args.idle_timeout:
            terminate_process_group(proc, log_f, "idle_timeout")
            return -124, "idle_timeout"

        time.sleep(args.watchdog_interval)


def selected_tasks(rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Tuple[Dict[str, str], str]]:
    by_name = {row["candidate"]: row for row in rows}
    if args.tasks:
        tasks = []
        for item in args.tasks.split(","):
            candidate, suite = item.split(":", 1)
            if candidate not in by_name:
                raise RuntimeError("unknown candidate in --tasks: " + candidate)
            if suite not in SUITE_WEATHER:
                raise RuntimeError("unknown suite in --tasks: " + suite)
            tasks.append((by_name[candidate], suite))
        return tasks

    if args.candidates:
        wanted = [x for x in args.candidates.split(",") if x]
    else:
        wanted = [row["candidate"] for row in rows]
    suites = [x for x in args.suites.split(",") if x]
    for suite in suites:
        if suite not in SUITE_WEATHER:
            raise RuntimeError("unknown suite: " + suite)
    return [(by_name[candidate], suite) for candidate in wanted for suite in suites]


def run_task(row: Dict[str, str], suite: str, target_rows: int, args: argparse.Namespace, task_index: int) -> bool:
    candidate = row["candidate"]
    mutation_type = row["mutation_type"]
    weather = SUITE_WEATHER[suite]
    cand_dir = os.path.join(args.output_root, candidate)
    os.makedirs(cand_dir, exist_ok=True)
    output_csv = os.path.join(cand_dir, "{}.csv".format(suite))
    log_path = os.path.join(cand_dir, "{}.log".format(suite))

    if data_rows(output_csv) >= target_rows:
        append_status(args.status_csv, [
            candidate, mutation_type, suite, weather, 0, "", now_str(), 0,
            data_rows(output_csv), target_rows, "skipped_done", output_csv, log_path,
        ])
        print("[SKIP] {} {} already has {} rows".format(candidate, suite, target_rows), flush=True)
        return True

    for attempt in range(1, args.max_attempts + 1):
        started = now_str()
        ensure_carla(args.port, args.carla_start_script, args.carla_wait)
        cmd = [
            PYTHON, EVALUATE,
            "--candidate-name", candidate,
            "--candidate-ckpt", row["staged_ckpt"],
            "--suite", suite,
            "--weather", weather,
            "--port", str(args.port),
            "--test-cases", args.test_cases,
            "--output-csv", output_csv,
            "--max-steps", str(args.max_steps),
            "--mutation-type", mutation_type,
            "--resume",
        ]
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.observer:
            observer_log = os.path.join(cand_dir, "observer_{}.log".format(suite))
            cmd.extend([
                "--observer",
                "--observer-web-host", args.observer_web_host,
                "--observer-web-port", str(args.observer_web_port),
                "--observer-log", observer_log,
            ])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MUTATION_INTENSITY", "1.0")
        print("[RUN] {} {} attempt {}/{} weather={}".format(
            candidate, suite, attempt, args.max_attempts, weather), flush=True)
        with open(log_path, "a") as log_f:
            log_f.write("\n[RUNNER] {} cmd={}\n".format(started, " ".join(cmd)))
            log_f.flush()
            returncode, watchdog_status = run_child_with_watchdog(
                cmd, env, log_f, output_csv, target_rows, args)

        rows = data_rows(output_csv)
        ended = now_str()
        if returncode == 0 and rows >= target_rows:
            status = "done"
        elif returncode == 0:
            status = "partial"
        elif watchdog_status != "exit":
            status = watchdog_status
        else:
            status = "failed"
        append_status(args.status_csv, [
            candidate, mutation_type, suite, weather, attempt, started, ended,
            returncode, rows, target_rows, status, output_csv, log_path,
        ])
        print("[STATUS] {} {} status={} rows={}/{} exit={}".format(
            candidate, suite, status, rows, target_rows, returncode), flush=True)
        if status == "done":
            open(output_csv + ".done", "a").close()
            return True
        if args.restart_carla_on_fail:
            restart_carla(args.port, args.carla_start_script, args.carla_wait)

    return data_rows(output_csv) >= target_rows


def main() -> int:
    args = parse_args()
    target_rows = args.limit if args.limit > 0 else read_case_count(args.test_cases)
    rows = read_manifest(args.manifest)
    tasks = selected_tasks(rows, args)
    ensure_status_header(args.status_csv)
    print("[CFG] target_rows={} tasks={} output_root={}".format(
        target_rows, len(tasks), args.output_root), flush=True)
    print("[CFG] watchdog_interval={} idle_timeout={} carla_missing_timeout={}".format(
        args.watchdog_interval, args.idle_timeout, args.carla_missing_timeout), flush=True)
    ok = True
    for idx, (row, suite) in enumerate(tasks):
        ok = run_task(row, suite, target_rows, args, idx) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

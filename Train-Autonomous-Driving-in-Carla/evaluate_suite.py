"""
Plan-06 evaluation suite: run a candidate (baseline/mutant) on N=500 test cases.

Suites: validation (ClearNoon) / sunny (ClearNoon) / rainy (MidRainyNoon) / foggy (Custom_Foggy)
Raw fail rule: episode does not reach final waypoint. Phase 6 computes relative_fail.
Mutant evaluation activates mutation hooks (mutation module exists only on mutation/runtime branch)

CSV schema: see plan-06 04-evaluation-spec.md section 5.1
"""
import os
import sys
import time
import csv
import json
import random
import argparse
import subprocess
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla
from simulation import settings as sim_settings
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from parameters import LATENT_DIM, ACTION_STD_INIT, EPISODE_LENGTH


EXPECTED_WEATHER_BY_SUITE = {
    "validation": "ClearNoon",
    "sunny": "ClearNoon",
    "rainy": "MidRainyNoon",
    "foggy": "Custom_Foggy",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-name", required=True, help="candidate label")
    p.add_argument("--candidate-ckpt", required=True, help=".pth state_dict path")
    p.add_argument("--suite", required=True, choices=["validation", "sunny", "rainy", "foggy"])
    p.add_argument("--weather", required=True,
                   help="CARLA WeatherParameters name (ClearNoon / MidRainyNoon / Custom_Foggy)")
    p.add_argument("--port", type=int, default=2002, help="evaluation port (training uses 2000)")
    p.add_argument("--test-cases", default="/root/autodl-tmp/eval/test_cases_v1.json")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--max-steps", type=int, default=EPISODE_LENGTH)
    p.add_argument("--mutation-type", default="none")
    p.add_argument("--town", default="Town07")
    p.add_argument("--limit", type=int, default=0, help="Optional: limit to first N cases (0 = all)")
    p.add_argument("--eval-seed", type=int, default=20260514, help="base seed for per-case Python RNGs")
    p.add_argument("--resume", action="store_true", help="Append to output CSV and skip completed case_id rows")
    p.add_argument("--allow-weather-override", action="store_true",
                   help="Allow suite/weather mismatch; default is strict for Phase 5")
    p.add_argument("--observer", action="store_true", help="Start tools/carla_pygame_observer.py during evaluation")
    p.add_argument("--observer-web-host", default="127.0.0.1", help="observer HTTP bind host")
    p.add_argument("--observer-web-port", type=int, default=8090, help="observer HTTP port")
    p.add_argument("--observer-max-fps", type=float, default=10.0, help="observer stream FPS cap")
    p.add_argument("--observer-log", default="", help="observer log path; default is output CSV dir/observer.log")
    return p.parse_args()


def build_weather(name):
    if name == "Custom_Foggy":
        return carla.WeatherParameters(
            cloudiness=80.0, precipitation=0.0, sun_altitude_angle=70.0,
            fog_density=80.0, fog_distance=10.0, fog_falloff=2.0
        )
    if hasattr(carla.WeatherParameters, name):
        return getattr(carla.WeatherParameters, name)
    raise ValueError("Unknown weather preset: " + name)


def validate_suite_weather(args):
    expected = EXPECTED_WEATHER_BY_SUITE.get(args.suite)
    if expected and args.weather != expected and not args.allow_weather_override:
        raise ValueError(
            "suite/weather mismatch: suite=%s expects weather=%s, got %s "
            "(pass --allow-weather-override only for an intentional ablation)"
            % (args.suite, expected, args.weather)
        )


def _parse_raw_fail(value):
    try:
        return int(float(value))
    except Exception:
        return None


def read_existing_results(output_csv, fieldnames):
    # Phase 5 长评估允许断点续跑；已有 CSV 只按完整 case_id 行计数。
    completed_case_ids = set()
    n_pass = 0
    n_fail = 0
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return completed_case_ids, n_pass, n_fail
    with open(output_csv, newline="") as csv_f:
        reader = csv.DictReader(csv_f)
        missing = [name for name in fieldnames if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError("existing CSV schema mismatch, missing fields: " + str(missing))
        for row in reader:
            case_id = row.get("case_id")
            if case_id is None or case_id == "" or case_id in completed_case_ids:
                continue
            completed_case_ids.add(case_id)
            raw_fail = _parse_raw_fail(row.get("raw_fail"))
            if raw_fail == 1:
                n_fail += 1
            elif raw_fail == 0:
                n_pass += 1
    return completed_case_ids, n_pass, n_fail


def open_output_csv(output_csv, fieldnames, resume):
    append = resume and os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    mode = "a" if append else "w"
    csv_f = open(output_csv, mode, newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
        csv_f.flush()
    return csv_f, writer


def start_observer(args):
    if not args.observer:
        return None, None
    observer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "carla_pygame_observer.py")
    if not os.path.exists(observer_path):
        raise RuntimeError("observer script not found: " + observer_path)
    log_path = args.observer_log or os.path.join(os.path.dirname(os.path.abspath(args.output_csv)), "observer.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_f = open(log_path, "a")
    cmd = [
        sys.executable, observer_path,
        "--host", sim_settings.HOST,
        "--port", str(args.port),
        "--web-host", args.observer_web_host,
        "--web-port", str(args.observer_web_port),
        "--max-fps", str(args.observer_max_fps),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    print("[OBSERVER] started pid=" + str(proc.pid) +
          " url=http://" + args.observer_web_host + ":" + str(args.observer_web_port) + "/" +
          " log=" + log_path)
    return proc, log_f


def stop_observer(proc, log_f):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        print("[OBSERVER] stopped")
    if log_f is not None:
        log_f.close()


def load_weights_into_agent(agent, weight_path):
    state = torch.load(weight_path, map_location="cpu")
    res_old = agent.old_policy.load_state_dict(state, strict=False)
    res_new = agent.policy.load_state_dict(state, strict=False)
    print("[LOAD] " + weight_path)
    print("[LOAD] old_policy missing=" + str(res_old.missing_keys) + " unexpected=" + str(res_old.unexpected_keys))
    print("[LOAD] policy     missing=" + str(res_new.missing_keys) + " unexpected=" + str(res_new.unexpected_keys))
    if res_old.unexpected_keys or res_new.unexpected_keys:
        raise RuntimeError("unexpected keys in checkpoint: " + str(res_old.unexpected_keys))
    allowed_missing = {"cov_var", "cov_mat"}
    bad = set(res_old.missing_keys) - allowed_missing
    if bad:
        raise RuntimeError("unacceptable missing keys: " + str(bad))
    for p in agent.old_policy.parameters():
        p.requires_grad = False
    for p in agent.policy.parameters():
        p.requires_grad = False


def main():
    args = parse_args()

    validate_suite_weather(args)
    os.environ["MUTATION_TYPE"] = args.mutation_type
    try:
        from mutation import config as _mutation_config
        _mutation_config.init()
    except ImportError:
        if args.mutation_type != "none":
            print("[WARN] mutation_type=" + args.mutation_type + " but mutation module not found; expected on mutation/runtime branch")

    sim_settings.PORT = args.port
    print("[CFG] port=" + str(args.port) + " town=" + args.town + " suite=" + args.suite + " weather=" + args.weather)
    print("[CFG] candidate=" + args.candidate_name + " ckpt=" + args.candidate_ckpt)
    print("[CFG] mutation_type=" + args.mutation_type)
    if args.observer:
        print("[CFG] observer=http://" + args.observer_web_host + ":" + str(args.observer_web_port) + "/")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.test_cases) as f:
        cases = json.load(f)
    if args.limit > 0:
        cases = cases[:args.limit]
    print("[CFG] Loaded " + str(len(cases)) + " test cases from " + args.test_cases)

    fieldnames = [
        "case_id", "candidate", "suite", "spawn_idx", "heading_offset_deg",
        "total_reward", "total_steps", "distance_m", "done_reason", "raw_fail",
        "progress_ratio", "final_waypoint_idx", "route_length", "wall_time_s", "mutation_type",
    ]
    t_global = datetime.now()

    if args.resume:
        completed_case_ids, n_pass, n_fail = read_existing_results(args.output_csv, fieldnames)
    else:
        completed_case_ids, n_pass, n_fail = set(), 0, 0
    cases_to_run = [c for c in cases if str(c.get("case_id")) not in completed_case_ids]
    print("[CFG] resume=" + str(args.resume) +
          " completed=" + str(len(completed_case_ids)) +
          " remaining=" + str(len(cases_to_run)))

    if not cases_to_run:
        tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
        print("[DONE] " + args.candidate_name + " on " + args.suite +
              ": pass=" + str(n_pass) + " fail=" + str(n_fail) + " TFR=" + ("%.3f" % tfr))
        print("[DONE] CSV: " + args.output_csv)
        print("[DONE] elapsed: 0.0 min")
        return

    client, world = ClientConnection(args.town).setup()
    weather = build_weather(args.weather)
    world.set_weather(weather)

    env = CarlaEnvironment(client, world, args.town, checkpoint_frequency=None)
    n_spawn_points = len(env.map.get_spawn_points())
    bad_cases = [c for c in cases if int(c.get("spawn_idx", -1)) < 0 or int(c.get("spawn_idx", -1)) >= n_spawn_points]
    if bad_cases:
        sample = bad_cases[:5]
        raise ValueError("test cases contain spawn_idx outside 0..%d: %s" % (n_spawn_points - 1, sample))
    print("[CFG] Town spawn points=" + str(n_spawn_points))
    encode = EncodeState(LATENT_DIM)
    agent = PPOAgent(args.town, ACTION_STD_INIT)
    load_weights_into_agent(agent, args.candidate_ckpt)

    observer_proc, observer_log = start_observer(args)
    csv_f, writer = open_output_csv(args.output_csv, fieldnames, args.resume)
    try:
        processed_new = 0

        for case in cases_to_run:
            t0 = time.time()
            case_seed = (args.eval_seed * 1000003 + int(case["case_id"])) & 0x7FFFFFFF
            random.seed(case_seed)
            np.random.seed(case_seed)
            torch.manual_seed(case_seed)
            env._eval_spawn_idx = case["spawn_idx"]
            env._eval_heading_offset_deg = case["heading_offset_deg"]

            obs = env.reset()
            if obs is None:
                writer.writerow({
                    "case_id": case["case_id"], "candidate": args.candidate_name,
                    "suite": args.suite, "spawn_idx": case["spawn_idx"],
                    "heading_offset_deg": round(case["heading_offset_deg"], 4),
                    "total_reward": 0.0, "total_steps": 0, "distance_m": 0.0,
                    "done_reason": "reset_failure", "raw_fail": 1,
                    "progress_ratio": 0.0, "final_waypoint_idx": 0, "route_length": 0,
                    "wall_time_s": round(time.time() - t0, 2),
                    "mutation_type": args.mutation_type,
                })
                csv_f.flush()
                n_fail += 1
                processed_new += 1
                if processed_new % 50 == 0 or processed_new == len(cases_to_run):
                    tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
                    elapsed = (datetime.now() - t_global).total_seconds() / 60.0
                    print("[PROGRESS] total=" + str(n_pass + n_fail) + "/" + str(len(cases)) +
                          " | new=" + str(processed_new) + "/" + str(len(cases_to_run)) +
                          " | pass=" + str(n_pass) + " fail=" + str(n_fail) +
                          " TFR=" + ("%.3f" % tfr) + " | elapsed=" + ("%.1f" % elapsed) + "min")
                continue

            obs = encode.process(obs)
            total_reward = 0.0
            steps = 0
            done = False
            done_reason = "max_steps_reached"
            info = None

            for step_i in range(args.max_steps):
                action = agent.get_action(obs, step_i, total_reward, done, train=False, deterministic=True)
                step_out = env.step(action)
                if step_out is None or step_out[0] is None:
                    done_reason = "step_failure"
                    break
                obs, reward, done, info = step_out
                obs = encode.process(obs)
                if torch.any(torch.isnan(obs)):
                    obs = torch.zeros_like(obs)
                total_reward += float(reward)
                steps += 1
                if done:
                    done_reason = env.get_last_done_reason() or "unknown_done"
                    break

            final_idx = getattr(env, "current_waypoint_index", 0)
            route_len = len(env.route_waypoints) if getattr(env, "route_waypoints", None) else 0
            raw_fail = 0 if done_reason == "route_completed" else 1
            progress_ratio = max(0.0, min(1.0, float(final_idx) / max(1, route_len - 1)))

            writer.writerow({
                "case_id": case["case_id"], "candidate": args.candidate_name,
                "suite": args.suite, "spawn_idx": case["spawn_idx"],
                "heading_offset_deg": round(case["heading_offset_deg"], 4),
                "total_reward": round(total_reward, 4), "total_steps": steps,
                "distance_m": round(info[0], 2) if info else 0.0,
                "done_reason": done_reason, "raw_fail": raw_fail,
                "progress_ratio": round(progress_ratio, 6),
                "final_waypoint_idx": final_idx, "route_length": route_len,
                "wall_time_s": round(time.time() - t0, 2),
                "mutation_type": args.mutation_type,
            })
            csv_f.flush()

            if raw_fail:
                n_fail += 1
            else:
                n_pass += 1
            processed_new += 1

            if processed_new % 50 == 0 or processed_new == len(cases_to_run):
                tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
                elapsed = (datetime.now() - t_global).total_seconds() / 60.0
                print("[PROGRESS] total=" + str(n_pass + n_fail) + "/" + str(len(cases)) +
                      " | new=" + str(processed_new) + "/" + str(len(cases_to_run)) +
                      " | pass=" + str(n_pass) + " fail=" + str(n_fail) +
                      " TFR=" + ("%.3f" % tfr) + " | elapsed=" + ("%.1f" % elapsed) + "min")
    finally:
        csv_f.close()
        stop_observer(observer_proc, observer_log)

    tfr = n_fail / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0.0
    elapsed = (datetime.now() - t_global).total_seconds() / 60.0
    print("[DONE] " + args.candidate_name + " on " + args.suite +
          ": pass=" + str(n_pass) + " fail=" + str(n_fail) + " TFR=" + ("%.3f" % tfr))
    print("[DONE] CSV: " + args.output_csv)
    print("[DONE] elapsed: " + ("%.1f" % elapsed) + " min")


if __name__ == "__main__":
    main()
